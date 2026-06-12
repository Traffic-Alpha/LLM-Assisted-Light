# LLM-Based Traffic Signal Control

This repository is a minimal, runnable LLM-based traffic signal control method
for single-intersection SUMO/TSHub scenarios.

It contains:

- `run_llm_tsc.py`: LLM controller with tool calls and event-aware observations
- `run_maxpressure.py`: max-pressure baseline on the same scenarios
- `llm_tsc/`: lightweight agent, tools, config, event wrapper, and baseline logic
- `traffic_env/`: SUMO/TSHub environment wrappers
- `scenarios/`: 3-way and 4-way SUMO scenarios plus event YAML files
- `tests/`: lightweight unit tests

No LangChain code is used.

## Install

Install TransSimHub and the small Python dependency set used by this project:

```bash
pip install openai pyyaml loguru gymnasium numpy pytest
```

Install SUMO/TSHub following the TransSimHub documentation.

Copy and edit the config template:

```bash
cp config.yaml.example config.yaml
```

Set `OPENAI_API_KEY` in `config.yaml` or via environment variable.

## Run LLM-TSC

```bash
python run_llm_tsc.py --scenario 4way --phase-num 4 --event-config scenarios/4way/events/accident_set1.yaml
```

The controller uses the LLM agent for every traffic-signal decision step.

Change the simulation horizon with `TOTAL_SIMULATION_TIME` in
`config.yaml.example` / `config.yaml`.

## Run MaxPressure

```bash
python run_maxpressure.py --scenario 4way --phase-num 4 --event-config scenarios/4way/events/accident_set1.yaml
```

## Events

Events live in:

```text
scenarios/<scenario>/events/*.yaml
```

Supported event types:

- `ACCIDENTS`: insert stopped obstacle vehicles
- `SPECIAL_VEHICLES`: schedule ambulance/police/rescue vehicles
- `SENSOR_FAILURES`: mask detector/movement occupancy

See [docs/event_config.md](docs/event_config.md) for details.

## About RL

RL code and trained models are intentionally not included. If you want an RL
controller, train it in a separate RL repository or workflow and add a separate
runner or controller module.

## Test

```bash
python -m pytest tests
```
