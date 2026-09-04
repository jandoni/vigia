"""Tier 0 — the camera registry and context gate.

Running five detectors on every frame of every camera is affordable in a demo
and not in a deployment. The gate is what makes the detector count a design
choice rather than a cost: each camera is registered once with a context, and
the context declares which hazards can physically occur there. A ridgeline
camera never runs the flood segmenter; a motorway camera never looks for people
in water.

DECLARATIVE, NOT LEARNED — and that is the decision, not an omission. We do not
build a classifier that inspects a frame and routes it. Such a classifier adds a
misrouting failure mode on the critical path, in exchange for information the
operator already has and can simply state. Every real deployment works this way:
Pano AI does not run drowning detection on a ridgeline camera. The published
evidence for the efficiency gain is cascaded gating in video pipelines, which
reports 5-13x compute reduction at minimal accuracy cost (NoScope,
arXiv 1703.02529; CaTDet).

The gate is therefore a lookup table with provenance, and its correctness is a
matter of the registry being right rather than of a model being accurate. That
trade is deliberate: a wrong entry here is visible, auditable and fixable by a
human reading a YAML file, whereas a misrouting classifier fails silently.

PRIVACY. The registry is also where retention policy lives, because retention is
a property of a camera's deployment rather than of a detector. See
`RetentionPolicy` and PLAN.md section 10: detection happens at the edge, only the
event and a single evidence frame leave the camera, and no code path exists that
could identify a person.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from vigia.types import Hazard


class Viewpoint(str, Enum):
    """How a camera looks at the world.

    This exists because a detector's training viewpoint is part of its
    operating envelope and nothing else in the system captured it. The flood
    segmenter is trained on ATLANTIS, which is ground-level and oblique
    photography of water bodies; run on straight-down UAV imagery it reports up
    to 0.397 water coverage on frames whose actual water is a narrow canal,
    tinting mown grass as water. It does not fail loudly — it returns a
    confident wrong number, which is the worst way for a life-safety component
    to be wrong.

    Declaring viewpoint turns that into a refusal at construction time.
    """

    GROUND = "ground"                  # eye level, looking across a scene
    OBLIQUE = "oblique"                # elevated and angled: CCTV, mast, hillside
    NADIR_AERIAL = "nadir_aerial"      # UAV or aircraft looking down


class CameraContext(str, Enum):
    """Where a camera is pointed, in the only terms the gate needs.

    String-valued so a registry file reads as plain YAML and so an unknown
    context fails loudly at load time rather than silently disabling a hazard.
    """

    FOREST = "forest"
    WILDLAND_INTERFACE = "wildland_interface"
    STREET = "street"
    URBAN_BLOCK = "urban_block"
    RIVER = "river"
    COAST = "coast"
    MOTORWAY = "motorway"
    EARTHQUAKE_ZONE = "earthquake_zone"
    UAV_SEARCH = "uav_search"


#: Which hazards can physically occur in each context.
#:
#: This is the whole of Tier 0's intelligence and it is intentionally small
#: enough to audit at a glance. The bias is toward including a hazard when it is
#: plausible: a missed hazard is a life-safety failure, while an unnecessary
#: detector costs only compute, which is what the gate was saving in the first
#: place. Where that bias was resisted it is noted.
CONTEXT_HAZARDS: dict[CameraContext, frozenset[Hazard]] = {
    # Ridgeline and landscape cameras — PyroNear's actual deployment envelope.
    CameraContext.FOREST: frozenset({Hazard.FIRE}),
    # Where settlement meets vegetation: fire plus the roads people flee along.
    CameraContext.WILDLAND_INTERFACE: frozenset({Hazard.FIRE, Hazard.TRAFFIC}),
    # The DANA case. Water rising through a street, vehicles caught in it, and
    # people trapped in the buildings alongside.
    CameraContext.STREET: frozenset({
        Hazard.FLOOD, Hazard.TRAFFIC, Hazard.BUILDING_ACCESS,
    }),
    CameraContext.URBAN_BLOCK: frozenset({
        Hazard.FLOOD, Hazard.BUILDING_ACCESS,
    }),
    # A watercourse: level and rate of rise, plus anyone swept into it.
    CameraContext.RIVER: frozenset({Hazard.FLOOD, Hazard.DROWNING}),
    CameraContext.COAST: frozenset({Hazard.DROWNING}),
    # Deliberately narrow. Flood is plausible on a motorway but the flood
    # detector is calibrated for street-level scenes, and claiming a hazard the
    # envelope does not cover is worse than not claiming it.
    CameraContext.MOTORWAY: frozenset({Hazard.TRAFFIC}),
    CameraContext.EARTHQUAKE_ZONE: frozenset({Hazard.BUILDING_ACCESS}),
    # Aerial search and rescue: the DRespNeT and SeaDronesSee envelopes, both
    # of which are UAV imagery rather than fixed mounts.
    CameraContext.UAV_SEARCH: frozenset({
        Hazard.BUILDING_ACCESS, Hazard.DROWNING,
    }),
}


#: Default viewpoint per context. Overridable per camera, because a context
#: describes what a camera watches and not where it is mounted.
CONTEXT_VIEWPOINT: dict[CameraContext, Viewpoint] = {
    CameraContext.FOREST: Viewpoint.OBLIQUE,
    CameraContext.WILDLAND_INTERFACE: Viewpoint.OBLIQUE,
    CameraContext.STREET: Viewpoint.OBLIQUE,
    CameraContext.URBAN_BLOCK: Viewpoint.OBLIQUE,
    CameraContext.RIVER: Viewpoint.OBLIQUE,
    CameraContext.COAST: Viewpoint.OBLIQUE,
    CameraContext.MOTORWAY: Viewpoint.OBLIQUE,
    CameraContext.EARTHQUAKE_ZONE: Viewpoint.NADIR_AERIAL,
    CameraContext.UAV_SEARCH: Viewpoint.NADIR_AERIAL,
}


@dataclass(frozen=True)
class RetentionPolicy:
    """What a camera is permitted to keep, and for how long.

    Defaults are the restrictive ones. PLAN.md section 10 commits to no raw
    video retention beyond a short rolling buffer, and to only the event plus a
    single evidence frame leaving the camera; those commitments are expressed
    here as data so a deployment cannot quietly exceed them by omission.
    """

    #: Seconds of decoded video held in memory for context around an event.
    #: Not written to disk by the pipeline under any setting.
    rolling_buffer_seconds: float = 10.0
    #: Attach one still frame to a confirmed event as evidence.
    evidence_frame: bool = True
    #: Never true in this codebase. Present so that an operator reading the
    #: registry can see the commitment stated rather than inferred.
    store_raw_video: bool = False

    def __post_init__(self) -> None:
        if self.store_raw_video:
            raise ValueError(
                "store_raw_video is not supported. VIGÍA retains no raw video: "
                "detection happens at the edge and only the event plus a single "
                "evidence frame leaves the camera (PLAN.md section 10). This is "
                "enforced here rather than left to configuration."
            )
        if self.rolling_buffer_seconds < 0:
            raise ValueError("rolling_buffer_seconds must not be negative")


@dataclass
class Camera:
    """One registered camera.

    `hazards` overrides the context when set, for the case the context map
    cannot express — a river camera that also happens to overlook a road. An
    override is recorded explicitly rather than by inventing a new context, so
    the registry stays readable and the exception stays visible.
    """

    camera_id: str
    source: str                       # file path, RTSP URL, or webcam index
    context: CameraContext
    name: str = ""
    hazards: Optional[frozenset[Hazard]] = None
    #: Overrides the context default. A river gauge camera on a mast is
    #: oblique; the same river filmed from a drone is not.
    viewpoint: Optional[Viewpoint] = None
    enabled: bool = True
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)
    notes: str = ""

    @property
    def active_hazards(self) -> frozenset[Hazard]:
        """The hazards this camera actually runs. This is the gate."""
        if self.hazards is not None:
            return self.hazards
        return CONTEXT_HAZARDS.get(self.context, frozenset())

    @property
    def active_viewpoint(self) -> Viewpoint:
        if self.viewpoint is not None:
            return self.viewpoint
        return CONTEXT_VIEWPOINT.get(self.context, Viewpoint.OBLIQUE)

    @property
    def overrides_context(self) -> bool:
        return self.hazards is not None

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "source": self.source,
            "context": self.context.value,
            "name": self.name,
            "hazards": sorted(h.value for h in self.active_hazards),
            "hazards_overridden": self.overrides_context,
            "viewpoint": self.active_viewpoint.value,
            "enabled": self.enabled,
            "notes": self.notes,
        }


class UnknownCamera(KeyError):
    """Raised rather than returning an empty hazard set.

    An unregistered camera must never be silently treated as "no hazards
    apply" — that would turn a typo into a camera that watches for nothing and
    reports success.
    """


class CameraRegistry:
    """The set of registered cameras, and the gate over them.

    Usage:
        registry = CameraRegistry.from_yaml("cameras.yaml")
        for hazard in registry.hazards_for("valencia_calle_01"):
            ...
    """

    def __init__(self, cameras: Optional[Iterable[Camera]] = None) -> None:
        self._cameras: dict[str, Camera] = {}
        for camera in cameras or ():
            self.add(camera)

    # ------------------------------------------------------------------ #

    def add(self, camera: Camera) -> None:
        if camera.camera_id in self._cameras:
            raise ValueError(f"duplicate camera_id: {camera.camera_id}")
        self._cameras[camera.camera_id] = camera

    def get(self, camera_id: str) -> Camera:
        try:
            return self._cameras[camera_id]
        except KeyError:
            raise UnknownCamera(
                f"camera {camera_id!r} is not registered. Register it before "
                f"use; an unknown camera is an error, not an empty gate."
            ) from None

    def hazards_for(self, camera_id: str) -> frozenset[Hazard]:
        """THE GATE. Which hazards this camera runs."""
        camera = self.get(camera_id)
        return camera.active_hazards if camera.enabled else frozenset()

    def runs(self, camera_id: str, hazard: Hazard) -> bool:
        return hazard in self.hazards_for(camera_id)

    @property
    def cameras(self) -> list[Camera]:
        return list(self._cameras.values())

    @property
    def enabled_cameras(self) -> list[Camera]:
        return [c for c in self._cameras.values() if c.enabled]

    def required_hazards(self) -> frozenset[Hazard]:
        """Every hazard any enabled camera needs.

        The pipeline loads exactly these detectors — a five-hazard system
        watching only forest cameras loads one model, not five.
        """
        needed: set[Hazard] = set()
        for camera in self.enabled_cameras:
            needed |= camera.active_hazards
        return frozenset(needed)

    def __len__(self) -> int:
        return len(self._cameras)

    def __contains__(self, camera_id: object) -> bool:
        return camera_id in self._cameras

    # ------------------------------------------------------------------ #

    def gate_summary(self) -> dict:
        """What the gate is actually saving, in detector-invocations per frame.

        The saving is the point of Tier 0, so it is measured rather than
        asserted. `ungated` is what running every detector on every camera
        would cost; `gated` is what the registry asks for.
        """
        all_hazards = len(Hazard)
        cameras = self.enabled_cameras
        ungated = len(cameras) * all_hazards
        gated = sum(len(c.active_hazards) for c in cameras)
        return {
            "cameras": len(cameras),
            "hazards_available": all_hazards,
            "detector_invocations_per_frame_ungated": ungated,
            "detector_invocations_per_frame_gated": gated,
            "reduction": round(1.0 - gated / ungated, 4) if ungated else 0.0,
            "models_loaded": sorted(h.value for h in self.required_hazards()),
            "by_context": {
                context.value: sum(1 for c in cameras if c.context is context)
                for context in CameraContext
                if any(c.context is context for c in cameras)
            },
        }

    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, data: dict) -> "CameraRegistry":
        """Build from a parsed registry file.

        Every failure here is loud. A registry that half-loads is worse than one
        that refuses to load: it produces a system that appears to be watching.
        """
        registry = cls()
        entries = data.get("cameras") or []
        if not isinstance(entries, list):
            raise ValueError("`cameras` must be a list")

        for index, entry in enumerate(entries):
            if "camera_id" not in entry:
                raise ValueError(f"camera at index {index} has no camera_id")
            camera_id = entry["camera_id"]

            if "source" not in entry:
                raise ValueError(f"camera {camera_id!r} has no source")

            raw_context = entry.get("context")
            try:
                context = CameraContext(raw_context)
            except ValueError:
                raise ValueError(
                    f"camera {camera_id!r} has unknown context {raw_context!r}. "
                    f"Known contexts: "
                    f"{', '.join(c.value for c in CameraContext)}"
                ) from None

            viewpoint = None
            if entry.get("viewpoint") is not None:
                try:
                    viewpoint = Viewpoint(entry["viewpoint"])
                except ValueError:
                    raise ValueError(
                        f"camera {camera_id!r} has unknown viewpoint "
                        f"{entry['viewpoint']!r}. Known: "
                        f"{', '.join(v.value for v in Viewpoint)}"
                    ) from None

            hazards = None
            if entry.get("hazards") is not None:
                try:
                    hazards = frozenset(Hazard(h) for h in entry["hazards"])
                except ValueError as exc:
                    raise ValueError(
                        f"camera {camera_id!r} lists an unknown hazard: {exc}"
                    ) from None

            retention_data = entry.get("retention") or {}
            retention = RetentionPolicy(
                rolling_buffer_seconds=float(
                    retention_data.get("rolling_buffer_seconds", 10.0)
                ),
                evidence_frame=bool(retention_data.get("evidence_frame", True)),
                store_raw_video=bool(retention_data.get("store_raw_video", False)),
            )

            registry.add(Camera(
                camera_id=camera_id,
                source=str(entry["source"]),
                context=context,
                name=entry.get("name", ""),
                hazards=hazards,
                viewpoint=viewpoint,
                enabled=bool(entry.get("enabled", True)),
                retention=retention,
                notes=entry.get("notes", ""),
            ))
        return registry

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CameraRegistry":
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no camera registry at {path}")
        return cls.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    def to_dict(self) -> dict:
        return {"cameras": [camera.to_dict() for camera in self.cameras]}
