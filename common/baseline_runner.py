"""Run baseline controllers with common scenarios, events, and outputs."""
import argparse

from common.config import load_config
from common.logger import logger
from common.runtime import (
    add_simulation_arguments, append_decision_log, build_env,
    snapshot_env_decision_state,
)


def run_baseline(method: str) -> None:
    parser = argparse.ArgumentParser(description=f"{method} traffic signal baseline")
    add_simulation_arguments(parser)
    if method == "fixtime":
        from methods.fixtime.controller import green_time_seconds
        parser.add_argument("--green-time", type=green_time_seconds, default=20,
                            help="Green seconds per phase; yellow time is additional (default: 20)")
    args = parser.parse_args()
    config = load_config(args.config)
    num_seconds = args.num_seconds or config["TOTAL_SIMULATION_TIME"]
    delta_time = args.green_time if method == "fixtime" else 5
    env = build_env(args, num_seconds, method, delta_time=delta_time)
    if method == "fixtime":
        from methods.fixtime.controller import FixedTimeController
        controller = FixedTimeController(args.phase_num, args.green_time)
    else:
        from methods.maxpressure.controller import MaxPressureController
        controller = MaxPressureController(args.phase_num)
    try:
        env.reset()
        sim_time = 0
        while sim_time < num_seconds:
            phase_id = controller.act() if method == "fixtime" else controller.act(env)
            append_decision_log(args.decision_log, {
                "sim_time": sim_time, "controller": method, "source": method,
                "phase_id": phase_id,
                **({"green_time": args.green_time} if method == "fixtime" else {}),
                **snapshot_env_decision_state(env),
            })
            _, _, truncated, done, info = env.step(phase_id)
            sim_time = info["step_time"]
            if done or truncated:
                break
    finally:
        env.close()
    logger.info("Finished {} at {}s", method, sim_time)
