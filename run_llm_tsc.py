'''
@Author: WANG Maonan
@Date: 2026-06-11 18:09:09
@Description: Minimal LLM-based traffic signal controller (LLM-Assisted Light).
@LastEditTime: 2026-06-12 23:01:03
'''
import re
import json
import argparse

from pathlib import Path
from typing import Any, Dict

from openai import OpenAI
from llm_tsc.logger import logger

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from llm_tsc.agent import SimplifiedReActAgent
from llm_tsc.config import load_config
from llm_tsc.prompts import PromptManager
from llm_tsc.tools_adapter import TSCToolsAdapter
from llm_tsc.tools_registry import ToolRegistry
from traffic_env.event_wrapper import create_event_wrapper
from traffic_env.llm_wrapper import LLMTSCEnvWrapper
from traffic_env.tsc_env import TrafficSignalEnv


def parse_phase_id(value) -> int | None:
    try:
        return int(str(value).split("-")[-1])
    except (TypeError, ValueError):
        return None


def parse_agent_response(response: str) -> Dict:
    data = {}
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                data = {}

    if not isinstance(data, dict):
        data = {}

    phase_id = parse_phase_id(data.get("decision"))

    return {
        "phase_id": phase_id,
        "explanation": data.get("explanation", response),
        "has_valid_decision": phase_id is not None,
    }


def get_traditional_fallback_phase(env, fallback_phase: int, phase_num: int) -> int:
    try:
        decision = env.get_traditional_decision()
    except Exception as e:
        logger.warning(
            "Failed to get traditional fallback decision; keeping phase {}: {}",
            fallback_phase,
            e,
        )
        return fallback_phase % phase_num

    phase_id = parse_phase_id(decision.get("recommended_phase"))
    if phase_id is None:
        logger.warning(
            "Traditional fallback returned invalid phase {}; keeping phase {}",
            decision.get("recommended_phase"),
            fallback_phase,
        )
        return fallback_phase % phase_num

    return phase_id % phase_num


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


def build_env(args):
    path_convert = get_abs_path(__file__)
    route_type = "vehicle"
    sumo_cfg = path_convert(f"./scenarios/{args.scenario}/env/{route_type}.sumocfg")
    net_file = path_convert(f"./scenarios/{args.scenario}/env/{args.scenario}.net.xml")
    trip_info = path_convert(f"./{args.scenario}_llm_tsc.tripinfo.xml")

    env = TrafficSignalEnv(
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        trip_info=trip_info,
        num_seconds=args.num_seconds,
        tls_id=args.tls_id,
        tls_action_type="choose_next_phase",
        use_gui=args.gui,
    )
    wrapped = LLMTSCEnvWrapper(
        env=env,
        tls_id=args.tls_id,
        phase_num=args.phase_num,
    )
    event_config = args.event_config or path_convert(
        f"./scenarios/{args.scenario}/events/default.yaml"
    )
    return create_event_wrapper(
        env=wrapped,
        env_name=args.scenario,
        event_config_name=event_config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal LLM-based TSC runner")
    parser.add_argument("--scenario", default="4way", help="Scenario name under scenarios/")
    parser.add_argument("--phase-num", type=int, default=4, help="Number of TLS phases")
    parser.add_argument("--event-config", default=None, help="Full event YAML path")
    parser.add_argument("--tls-id", default="J1", help="Traffic light id")
    parser.add_argument("--num-seconds", type=int, default=500, help="Simulation horizon")
    parser.add_argument("--decision-log", default=None, help="Optional JSONL path for per-decision logs")
    parser.add_argument("--raw-response-log", default=None, help="Optional JSONL path for raw LLM responses")
    parser.add_argument("--gui", action="store_true", help="Run SUMO with GUI")
    args = parser.parse_args()

    path_convert = get_abs_path(__file__)
    set_logger(path_convert("./"))

    config = load_config()
    if not config.validate():
        return

    client = OpenAI(
        api_key=config["OPENAI_API_KEY"],
        base_url=config["OPENAI_API_BASE"],
        timeout=60.0,
    )

    env = build_env(args)
    tools = ToolRegistry().register_from_object(TSCToolsAdapter(env))
    agent = SimplifiedReActAgent(
        llm_client=client,
        tools_dict=tools,
        system_prompt=PromptManager.get_system_prompt(),
        max_iterations=config["MAX_ITERATIONS"],
        model=config["OPENAI_API_MODEL"],
        temperature=config["OPENAI_TEMPERATURE"],
        verbose=config["LLM_VERBOSE"],
    )

    done = False
    sim_time = 0
    phase_id = 0
    llm_decisions = 0
    traditional_fallbacks = 0
    total_time = config["TOTAL_SIMULATION_TIME"]

    env.reset()
    while not done:
        llm_decisions += 1
        message = PromptManager.get_agent_message(
            sim_step=sim_time,
            last_step_action=phase_id,
        )
        response = agent.run(message, reset_history=True)
        append_decision_log(
            args.raw_response_log,
            {
                "sim_time": sim_time,
                "response_type": "initial",
                "raw_response": response,
            },
        )
        parsed = parse_agent_response(response)
        if not parsed["has_valid_decision"]:
            logger.warning("Failed to parse LLM phase decision; requesting JSON reformat")
            response = agent.run(PromptManager.get_error_handling_prompt(), reset_history=False)
            append_decision_log(
                args.raw_response_log,
                {
                    "sim_time": sim_time,
                    "response_type": "reformat",
                    "raw_response": response,
                },
            )
            parsed = parse_agent_response(response)

        if parsed["has_valid_decision"]:
            phase_id = parsed["phase_id"] % args.phase_num
            decision_source = "llm"
        else:
            traditional_fallbacks += 1
            phase_id = get_traditional_fallback_phase(
                env=env,
                fallback_phase=phase_id,
                phase_num=args.phase_num,
            )
            logger.warning(
                "Failed to parse LLM phase decision; using traditional fallback phase {}",
                phase_id,
            )
            decision_source = "traditional_fallback"
        explanation = parsed["explanation"]
        env_state = snapshot_env_decision_state(env)
        append_decision_log(
            args.decision_log,
            {
                "sim_time": sim_time,
                "controller": "llm_tsc",
                "source": decision_source,
                "phase_id": phase_id,
                "explanation": explanation,
                **env_state,
            },
        )
        logger.info(
            "Decision at {}s: source={}, phase={}, traditional={}",
            sim_time,
            decision_source,
            phase_id,
            env_state.get("traditional_decision", {}).get("recommended_phase"),
        )

        _, _, _, done, info = env.step(action=phase_id)
        sim_time = info["step_time"]

        if sim_time >= total_time:
            break

    env.close()
    logger.info(
        "Finished simulation with {} LLM decisions and {} traditional fallbacks",
        llm_decisions,
        traditional_fallbacks,
    )


if __name__ == "__main__":
    main()
