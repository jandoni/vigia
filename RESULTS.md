# VIGÍA — frozen results

Every figure below is checked against the results file named beside it by `tools/freeze_results.py`, which fails the build if the two disagree. Quote from this table rather than transcribing from JSON.

| model | figure | value | source |
|---|---|---|---|
| `fire.pyronear_yolo11s_sensitive` | box_level.precision | **0.4472** | `eval/results/fire_pyro_sdis_val_wide.json` |
| `fire.pyronear_yolo11s_sensitive` | box_level.recall | **0.902** | `eval/results/fire_pyro_sdis_val_wide.json` |
| `fire.pyronear_yolo11s_sensitive` | image_level.false_alarm_rate | **0.706** | `eval/results/fire_pyro_sdis_val_wide.json` |
| `flood.atlantis_deeplabv3` | test_split.water_iou | **0.7633** | `eval/results/flood.json` |
| `flood.atlantis_deeplabv3` | test_split.precision | **0.8799** | `eval/results/flood.json` |
| `flood.atlantis_deeplabv3` | test_split.recall | **0.852** | `eval/results/flood.json` |
| `flood.atlantis_deeplabv3` | test_split.f1 | **0.8657** | `eval/results/flood.json` |
| `flood.atlantis_deeplabv3` | test_split.pixel_accuracy | **0.9319** | `eval/results/flood.json` |
| `flood.atlantis_deeplabv3` | test_split.median_latency_ms | **21.8** | `eval/results/flood.json` |
| `person_in_water.seadronessee_rfdetr` | box_level.precision | **0.9199** | `eval/results/person_in_water.json` |
| `person_in_water.seadronessee_rfdetr` | box_level.recall | **0.9146** | `eval/results/person_in_water.json` |
| `person_in_water.seadronessee_rfdetr` | box_level.f1 | **0.9172** | `eval/results/person_in_water.json` |
| `person_in_water.seadronessee_rfdetr` | image_level.false_alarm_rate | **0.0222** | `eval/results/person_in_water.json` |
| `person_in_water.seadronessee_rfdetr` | ground_truth_swimmer_instances | **1218** | `eval/results/person_in_water.json` |
| `building_access.drespnet_yolov8_drn` | test_split.box_level.precision | **0.785** | `eval/results/building_access.json` |
| `building_access.drespnet_yolov8_drn` | test_split.box_level.recall | **0.942** | `eval/results/building_access.json` |
| `building_access.drespnet_yolov8_drn` | test_split.box_level.f1 | **0.856** | `eval/results/building_access.json` |
| `building_access.drespnet_yolov8_drn` | test_split.image_level.recall | **1.0** | `eval/results/building_access.json` |
| `building_access.drespnet_yolov8_drn` | test_split.image_level.false_alarm_rate | **0.083** | `eval/results/building_access.json` |
| `building_access.drespnet_yolov8_drn` | test_split.ground_truth_civilian_instances | **291** | `eval/results/building_access.json` |
| `building_access.drespnet_yolov8_drn` | valid_split.box_level.recall | **0.87** | `eval/results/building_access.json` |
| `flood_aerial.floodnet_deeplabv3` | val_flooded_water_iou | **0.4734** | `eval/results/flood_aerial.json` |
| `flood_aerial.floodnet_deeplabv3` | val_overall_water_iou | **0.7021** | `eval/results/flood_aerial.json` |
| `flood_aerial.floodnet_deeplabv3` | val_pixel_accuracy | **0.9568** | `eval/results/flood_aerial.json` |
| `flood_aerial.floodnet_deeplabv3` | best_epoch | **25** | `eval/results/flood_aerial.json` |

## Not measured

- `traffic.enos_yolo11x` — not_yet_measured. Integrated and running behind the validator — it supplied the second hazard in the cross-hazard suppression comparison (eval/run_multihazard.py, 88.7% suppression against wildfire's 86.2%, same cascade.py). `measured` stays not_yet_measured because that experiment measures the VALIDATOR, not this detector: no precision/recall for the accident class against held-out labels has been run. Retained explicitly as the generality experiment rather than as a disaster hazard.
