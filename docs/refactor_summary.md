# Refactor Summary

The repository has been reduced to a minimal LLM-based TSC method.

## Final Structure

```text
llm_tsc/          Lightweight LLM agent, tools, event config, baseline logic
traffic_env/      SUMO/TSHub environment and wrappers
scenarios/        3way/4way SUMO scenarios and event YAML files
tests/            Lightweight tests
run_llm_tsc.py    Main LLM-based controller
run_maxpressure.py Max-pressure baseline
```

## Removed

- LangChain agent/prompt code
- RL training, evaluation, saved models, and stable-baselines wrappers
- Legacy launchers from the old prototype
- SQLite memory/similarity utilities from the older prototype

## Kept

- OpenAI function-calling ReAct loop implemented locally
- Event wrapper for accidents, special vehicles, and sensor failures
- Max-pressure baseline as a separate runner

## Commands

```bash
python run_llm_tsc.py --scenario 4way --phase-num 4 --event-config scenarios/4way/events/accident_set1.yaml
python run_maxpressure.py --scenario 4way --phase-num 4 --event-config scenarios/4way/events/accident_set1.yaml
python -m pytest tests
```

RL is intentionally external now. To use RL again, train it in a separate
project and add a separate runner or controller module.
