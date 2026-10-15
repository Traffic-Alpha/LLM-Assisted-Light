"""Cycle through signal phases with 20 seconds of green by default."""
import argparse

from loguru import logger

from la_light.config import load_config
from la_light.runtime import (
    add_simulation_arguments, append_decision_log, build_env,
    snapshot_env_decision_state,
)


def green_time_seconds(value: str) -> int:
    duration = int(value)
    if not 3 < duration < 3600:
        raise argparse.ArgumentTypeError("green time must be between 4 and 3599 seconds")
    return duration


def main() -> None:
    parser = argparse.ArgumentParser(description="fixtime traffic signal baseline")
    add_simulation_arguments(parser)
    parser.add_argument("--green-time", type=green_time_seconds, default=20,
                        help="Green seconds per phase; yellow time is additional (default: 20)")
    args = parser.parse_args()
    config = load_config(args.config)
    num_seconds = args.num_seconds or config["TOTAL_SIMULATION_TIME"]
    env = build_env(args, num_seconds, "fixtime", delta_time=args.green_time)
    try:
        env.reset()
        sim_time = 0
        phase_id = 0
        while True:
            append_decision_log(args.decision_log, {
                "sim_time": sim_time, "controller": "fixtime", "source": "fixtime",
                "phase_id": phase_id, "green_time": args.green_time,
                **snapshot_env_decision_state(env),
            })
            _, done, truncated, info = env.step(phase_id)
            sim_time = info["step_time"]
            if done or truncated:
                break
            phase_id = (phase_id + 1) % args.phase_num
    finally:
        env.close()
    logger.info("Finished fixtime at {}s: {}", sim_time, info["termination_reason"])


if __name__ == "__main__":
    main()
