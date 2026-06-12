# Event Config Guide

Event YAML files live under:

```text
scenarios/<scenario>/events/*.yaml
```

Use them with:

```bash
python run_llm_tsc.py --scenario 4way --phase-num 4 --event-config scenarios/4way/events/accident_set1.yaml
python run_maxpressure.py --scenario 4way --phase-num 4 --event-config scenarios/4way/events/accident_set1.yaml
```

`--event-config` must be an existing YAML path. Short names such as
`accident_set1` are intentionally not resolved automatically.

## YAML Format

```yaml
ACCIDENTS:
  - id: accident_01
    depart_time: 150
    edge_id: "E1"
    lane_index: 0
    position: 60.0
    type: "parked_car"
    duration: 120

SPECIAL_VEHICLES:
  - id: ambulance_01
    vehicle_type: "ambulance"
    depart_time: 220
    route: ["E0", "E1", "E2"]
    priority: 1

SENSOR_FAILURES:
  - id: sensor_fail_E2
    depart_time: 180
    detector_id: "E2--s"
    duration: 60
```

## Runtime Behavior

- Accidents are inserted as stopped SUMO vehicles at `depart_time`.
- Special vehicles are scheduled on their configured route.
- Sensor failures mask the configured detector/movement occupancy.

The wrapper applies events automatically in `reset()` and `step()`.

## Programmatic Use

```python
from traffic_env import create_event_wrapper
from traffic_env.llm_wrapper import LLMTSCEnvWrapper
from traffic_env.tsc_env import TrafficSignalEnv

env = TrafficSignalEnv(...)
env = LLMTSCEnvWrapper(env=env, tls_id="J1", phase_num=4)
env = create_event_wrapper(
    env,
    env_name="4way",
    event_config_name="scenarios/4way/events/accident_set1.yaml",
)
```
