"""Shared environment construction and decision logs for all controllers."""
import argparse
import json
import re
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="Model/run YAML config (default: YAML_CONFIG_PATH or config.yaml)")
    parser.add_argument("--scenario", choices=("3way", "4way"), default="4way")
    parser.add_argument("--phase-num", type=positive_int, default=None,
                        help="Number of green phases (default: infer from network)")
    parser.add_argument("--event-config", required=True, help="Event YAML path")
    parser.add_argument("--tls-id", default="J1")
    parser.add_argument("--num-seconds", type=positive_int, default=None,
                        help="Optional simulation time limit; default: run until traffic and events finish")
    parser.add_argument("--trip-info", default=None, help="Tripinfo XML output path")
    parser.add_argument("--decision-log", default=None, help="Decision JSONL output path")
    parser.add_argument("--gui", action="store_true", help="Run SUMO with GUI")


def parse_phase_id(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def snapshot_env_decision_state(env) -> dict[str, Any]:
    state = {}
    for key, method in (("traditional_decision", "get_traditional_decision"),
                        ("junction_situation", "get_junction_situation")):
        try:
            state[key] = getattr(env, method)()
        except Exception as exc:
            state[key + "_error"] = str(exc)
    return state


def append_decision_log(log_path: str | None, record: dict[str, Any]) -> None:
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_env(args, num_seconds: int | None, method: str, *, delta_time: int = 5):
    from la_light.env.base_tsc_wrapper import BaseTSCEnvWrapper
    from la_light.env.event_wrapper import create_event_wrapper
    from la_light.env.tsc_env import TrafficSignalEnv

    scenario_dir = PROJECT_ROOT / "scenarios" / args.scenario
    net_file = scenario_dir / "env" / f"{args.scenario}.net.xml"
    logic = next((node for node in ET.parse(net_file).getroot().findall("tlLogic")
                  if node.get("id") == args.tls_id), None)
    if logic is None:
        raise ValueError(f"Traffic light {args.tls_id!r} not found in {net_file}")
    phase_num = sum("G" in phase.get("state", "") for phase in logic.findall("phase"))
    if args.phase_num is not None and args.phase_num != phase_num:
        raise ValueError(f"{args.scenario} has {phase_num} green phases, got --phase-num {args.phase_num}")
    args.phase_num = phase_num
    trip_info = Path(args.trip_info) if args.trip_info else PROJECT_ROOT / f"{args.scenario}_{method}.tripinfo.xml"
    trip_info.parent.mkdir(parents=True, exist_ok=True)
    env = TrafficSignalEnv(
        sumo_cfg=str(scenario_dir / "env" / "vehicle.sumocfg"),
        net_file=str(net_file), trip_info=str(trip_info.resolve()),
        num_seconds=num_seconds, tls_id=args.tls_id,
        tls_action_type="choose_next_phase", use_gui=args.gui, delta_time=delta_time,
    )
    wrapped = BaseTSCEnvWrapper(env, tls_id=args.tls_id, phase_num=phase_num)
    return create_event_wrapper(wrapped, args.scenario, args.event_config)
