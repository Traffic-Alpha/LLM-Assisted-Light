'''
@Author: WANG Maonan
@Date: 2026-06-11 18:09:09
@Description: Run the max-pressure baseline on the same scenarios/events.
--> python run_maxpressure.py --scenario 4way --event-config scenarios/4way/events/accident_set1.yaml --gui
@LastEditTime: 2026-06-12 00:51:46
'''
import argparse
import json
from pathlib import Path
from typing import Any, Dict

from llm_tsc.logger import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from traffic_env.base_tsc_wrapper import BaseTSCEnvWrapper
from traffic_env.event_wrapper import create_event_wrapper
from traffic_env.tsc_env import TrafficSignalEnv


def parse_phase_id(value: Any) -> int | None:
    try:
        return int(str(value).split("-")[-1])
    except (TypeError, ValueError):
        return None


def snapshot_env_decision_state(env) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    try:
        state["traditional_decision"] = env.get_traditional_decision()
    except Exception as e:
        state["traditional_error"] = str(e)

    try:
        state["junction_situation"] = env.get_junction_situation()
    except Exception as e:
        state["junction_error"] = str(e)

    return state


def append_decision_log(log_path: str | None, record: Dict[str, Any]) -> None:
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Max-pressure TSC baseline")
    parser.add_argument("--scenario", default="4way", help="Scenario name under scenarios/")
    parser.add_argument("--phase-num", type=int, default=4, help="Number of TLS phases")
    parser.add_argument("--event-config", default=None, help="Full event YAML path")
    parser.add_argument("--tls-id", default="J1", help="Traffic light id")
    parser.add_argument("--num-seconds", type=int, default=500, help="Simulation horizon")
    parser.add_argument("--decision-log", default=None, help="Optional JSONL path for per-decision logs")
    parser.add_argument("--gui", action="store_true", help="Run SUMO with GUI")
    args = parser.parse_args()

    path_convert = get_abs_path(__file__)
    set_logger(path_convert("./"))
    event_config = args.event_config or path_convert(
        f"./scenarios/{args.scenario}/events/default.yaml"
    ) # 默认的 event 配置文件路径为 scenarios/{scenario}/events/default.yaml

    env = TrafficSignalEnv(
        sumo_cfg=path_convert(f"./scenarios/{args.scenario}/env/vehicle.sumocfg"),
        net_file=path_convert(f"./scenarios/{args.scenario}/env/{args.scenario}.net.xml"),
        trip_info=path_convert(f"./{args.scenario}_maxpressure.tripinfo.xml"),
        num_seconds=args.num_seconds,
        tls_id=args.tls_id,
        tls_action_type="choose_next_phase",
        use_gui=args.gui,
    )
    wrapped = BaseTSCEnvWrapper(env=env, tls_id=args.tls_id, phase_num=args.phase_num)
    wrapped = create_event_wrapper(wrapped, args.scenario, event_config)

    done = False
    sim_time = 0
    phase_id = 0
    wrapped.reset()

    while not done and sim_time < args.num_seconds:
        env_state = snapshot_env_decision_state(wrapped)
        recommended_phase = (
            env_state.get("traditional_decision", {}).get("recommended_phase")
        )
        parsed_phase = parse_phase_id(recommended_phase)
        if parsed_phase is not None:
            phase_id = parsed_phase
        phase_id %= args.phase_num
        append_decision_log(
            args.decision_log,
            {
                "sim_time": sim_time,
                "controller": "maxpressure",
                "source": "maxpressure",
                "phase_id": phase_id,
                **env_state,
            },
        )
        logger.info(
            "Decision at {}s: phase={}, traditional={}",
            sim_time,
            phase_id,
            env_state.get("traditional_decision", {}).get("recommended_phase"),
        )
        _, _, _, done, info = wrapped.step(phase_id)
        sim_time = info["step_time"]

    wrapped.close()
    logger.info("Finished max-pressure baseline at {}s", sim_time)


if __name__ == "__main__":
    main()
