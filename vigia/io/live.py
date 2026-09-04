"""Public live cameras — real infrastructure, watched in real time.

VIGÍA's whole argument is that the cameras already exist. A demonstration that
only ever replays a curated clip does not make that argument; it asserts it.
This module points the pipeline at cameras that are genuinely live right now,
run by other people, for their own reasons.

WHY THESE TWO SOURCES AND NOT A BEACH WEBCAM
--------------------------------------------
The catalogue is short on purpose. A camera earns a place here only if it
passes the same two gates every demonstration clip passes (see
clips/MANIFEST.json), plus a third that only applies to live feeds:

  1. LICENCE. USGS imagery is a work of the United States Government and
     carries no restriction at all. HPWREN publishes its camera imagery for
     public use subject to attribution, which is a condition we can meet and
     do meet, on the frame. Most attractive live streams — beach cameras,
     resort cameras, city-centre webcams — are operated commercially and their
     terms forbid exactly this kind of reuse. Not being able to find the
     terms is a reason to leave a camera out, not a reason to assume.

  2. VIEWPOINT. A camera may only be paired with a detector whose declared
     operating envelope contains it (vigia/registry.py). This rules out the
     obvious crowd-pleaser: a beach webcam looking along the sand is a
     GROUND-level view, and the people-in-water detector is trained on
     SeaDronesSee, which is aerial and elevated imagery from 5-260 m. Pointing
     it at a beach camera does not fail loudly — it returns a confident wrong
     answer, which is the failure mode this project exists to refuse. The
     pipeline would reject the pairing at construction time, and rightly.

  3. CADENCE, STATED HONESTLY. These cameras sample every few minutes, not at
     video rate. That is not a limitation of the demonstration — it is the
     regime real municipal and hydrological cameras run in, and it is why the
     flood trend detector is sample-based rather than time-based (a fixed
     300-second window once failed outright on a camera sampled every fifteen
     minutes; see vigia/flood/level.py). The interval is declared per camera
     and shown in the interface, so nobody watches a still image wondering
     whether the system has crashed.

Consequence worth stating plainly: fire and flood have live public cameras
here; people-in-water and building access do not, because no public live
aerial feed exists that we may use. Those two remain recorded-clip hazards,
and the interface says so rather than quietly showing less.
"""

from __future__ import annotations

import datetime as dt
import logging
import urllib.request
from dataclasses import dataclass
from typing import Optional
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

USER_AGENT = "VIGIA/0.1 (research prototype; multi-hazard camera detection)"

#: USGS "NIMS" camera network. The catalogue API lists ~1,300 cameras; images
#: land in a public S3 bucket keyed by camera and capture time.
USGS_CATALOGUE_API = "https://api.waterdata.usgs.gov/nims/cameras"
USGS_BUCKET = "https://usgs-nims-images.s3.amazonaws.com/"
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

#: HPWREN's real-time mirror. The same network that produced FIgLib, which is
#: where this project's fire evaluation sequences come from.
HPWREN_REALTIME = "https://cdn.hpwren.ucsd.edu/RT/{camera}.jpg"


@dataclass(frozen=True)
class LiveCamera:
    """One public camera, with everything needed to use it honestly.

    `interval_seconds` is the publisher's own update cadence, not a polling
    choice of ours: fetching faster than the camera updates returns the same
    image and wastes someone else's bandwidth.
    """

    camera_id: str
    source: str                 # usgs://<camId> or hpwren://<camera>
    name: str
    place: str
    context: str                # a CameraContext value — Tier 0 still decides
    viewpoint: str              # a Viewpoint value — enforced before any model loads
    hazard: str
    licence: str
    attribution: str
    interval_seconds: int
    home_url: str

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "place": self.place,
            "context": self.context,
            "viewpoint": self.viewpoint,
            "hazard": self.hazard,
            "licence": self.licence,
            "attribution": self.attribution,
            "interval_seconds": self.interval_seconds,
            "home_url": self.home_url,
            "live": True,
        }


#: The curated set. Deliberately small — every entry was checked against the
#: three gates in this module's docstring, and a camera whose terms could not
#: be established was left out rather than included hopefully.
CATALOGUE: tuple[LiveCamera, ...] = (
    LiveCamera(
        camera_id="usgs_asheville",
        source="usgs://NC_French_Broad_River_at_Asheville",
        name="French Broad River at Asheville",
        place="North Carolina, USA",
        context="river",
        viewpoint="oblique",
        hazard="flood",
        licence="Public domain — work of the United States Government",
        attribution="USGS streamgage camera, French Broad River at Asheville NC.",
        interval_seconds=300,
        home_url="https://apps.usgs.gov/hivis/",
    ),
    LiveCamera(
        camera_id="usgs_pittsburgh",
        source="usgs://PA_Monongahela_R_at_Point_State_Park_at_Pittsburgh",
        name="Monongahela River at Point State Park",
        place="Pittsburgh, Pennsylvania, USA",
        context="river",
        viewpoint="oblique",
        hazard="flood",
        licence="Public domain — work of the United States Government",
        attribution=("USGS streamgage camera, Monongahela River at Point State "
                     "Park, Pittsburgh PA."),
        interval_seconds=300,
        home_url="https://apps.usgs.gov/hivis/",
    ),
    LiveCamera(
        camera_id="hpwren_rm_west",
        source="hpwren://rm-w-mobo-c",
        name="Red Mountain, looking west",
        place="San Diego County, California, USA",
        context="forest",
        viewpoint="ground",
        hazard="fire",
        licence="HPWREN imagery — attribution required",
        attribution=("HPWREN, University of California San Diego "
                     "(hpwren.ucsd.edu)."),
        interval_seconds=60,
        home_url="https://hpwren.ucsd.edu/cameras/",
    ),
    LiveCamera(
        camera_id="hpwren_high_point",
        source="hpwren://hp-n-mobo-c",
        name="High Point, looking north",
        place="Palomar Mountain, California, USA",
        context="forest",
        viewpoint="ground",
        hazard="fire",
        licence="HPWREN imagery — attribution required",
        attribution=("HPWREN, University of California San Diego "
                     "(hpwren.ucsd.edu)."),
        interval_seconds=60,
        home_url="https://hpwren.ucsd.edu/cameras/",
    ),
)

BY_ID = {camera.camera_id: camera for camera in CATALOGUE}
BY_SOURCE = {camera.source: camera for camera in CATALOGUE}


# --------------------------------------------------------------------------- #
# Resolving "the current image" for each network
# --------------------------------------------------------------------------- #

def looks_live_source(source: str) -> bool:
    """True for a source this module knows how to poll."""
    text = str(source)
    return text.startswith(("usgs://", "hpwren://")) or (
        text.startswith(("http://", "https://"))
        and text.rsplit("?", 1)[0].lower().endswith((".jpg", ".jpeg", ".png"))
    )


def _http_get(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _usgs_latest_key(camera: str, *, days_back: int = 3) -> Optional[str]:
    """Newest image key for a USGS camera, or None if it has been quiet.

    Keys embed the capture time and sort chronologically, so narrowing the S3
    listing to one UTC day and taking the last key is both correct and cheap —
    the alternative, paging the camera's whole history, would fetch tens of
    thousands of keys to answer 'what is the latest'.
    """
    for back in range(days_back):
        day = (dt.datetime.now(dt.timezone.utc)
               - dt.timedelta(days=back)).strftime("%Y-%m-%d")
        url = (f"{USGS_BUCKET}?list-type=2"
               f"&prefix=720/{camera}/{camera}___{day}&max-keys=1000")
        try:
            tree = ElementTree.fromstring(_http_get(url))
        except Exception as exc:                      # network, DNS, throttling
            logger.warning("USGS listing failed for %s: %s", camera, exc)
            return None
        keys = [node.text for node in tree.findall(".//s3:Contents/s3:Key", _S3_NS)
                if node.text and node.text.endswith(".jpg")]
        if keys:
            return keys[-1]
    return None


def resolve_image_url(source: str) -> Optional[str]:
    """The URL of the CURRENT image for a live source.

    HPWREN publishes at a stable path that always holds the latest frame, so
    resolution is free. USGS publishes immutable timestamped objects, so the
    latest one has to be looked up — which is also what gives us the capture
    time for free, printed in the interface so a stale camera is visible as
    stale rather than as a system that has stopped.
    """
    text = str(source)
    if text.startswith("hpwren://"):
        return HPWREN_REALTIME.format(camera=text[len("hpwren://"):])
    if text.startswith("usgs://"):
        key = _usgs_latest_key(text[len("usgs://"):])
        return USGS_BUCKET + key if key else None
    if text.startswith(("http://", "https://")):
        return text
    return None


def capture_time(url: str) -> Optional[dt.datetime]:
    """Capture time parsed out of a USGS key, when the URL carries one."""
    import re

    match = re.search(r"___(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})Z", url)
    if not match:
        return None
    date, hour, minute, second = match.groups()
    return dt.datetime.fromisoformat(f"{date}T{hour}:{minute}:{second}").replace(
        tzinfo=dt.timezone.utc)


def fetch_image(url: str, timeout: float = 30.0):
    """Decode the image at `url`, or return None if anything goes wrong.

    Returning None rather than raising is deliberate: a public camera going
    offline for ten minutes is an ordinary event, not an error worth taking a
    running pipeline down for.
    """
    import cv2
    import numpy as np

    try:
        data = _http_get(url, timeout=timeout)
    except Exception as exc:
        logger.warning("live fetch failed for %s: %s", url, exc)
        return None
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        logger.warning("live fetch decoded to nothing: %s", url)
    return image
