"""Choose the same MP recommendation supplied to the LLM controller."""
import argparse

from loguru import logger

from la_light.config import load_config
from la_light.runtime import (
    add_simulation_arguments, append_decision_log, build_env,
    parse_phase_id, snapshot_env_decision_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="maxpressure traffic signal baseline")
    add_simulation_arguments(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    num_seconds = args.num_seconds or config["TOTAL_SIMULATION_TIME"]
    env = build_env(args, num_seconds, "maxpressure")
    try:
        env.reset()
        sim_time = 0
        phase_id = 0
        while True:
            recommended = parse_phase_id(env.get_traditional_decision().get("recommended_phase"))
            if recommended is not None:
                phase_id = recommended % args.phase_num
            append_decision_log(args.decision_log, {
                "sim_time": sim_time, "controller": "maxpressure", "source": "maxpressure",
                "phase_id": phase_id,
                **snapshot_env_decision_state(env),
            })
            _, done, truncated, info = env.step(phase_id)
            sim_time = info["step_time"]
            if done or truncated:
                break
    finally:
        env.close()
    logger.info("Finished maxpressure at {}s: {}", sim_time, info["termination_reason"])


if __name__ == "__main__":
    main()
