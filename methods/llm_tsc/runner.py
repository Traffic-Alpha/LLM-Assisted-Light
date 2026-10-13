'''
@Author: WANG Maonan
@Date: 2026-06-11 18:09:09
@Description: LLM-based traffic signal controller (LLM-Assisted Light).
--> python run_llm_tsc.py --scenario 4way --event-config scenarios/4way/events/special_vehicles_accident.yaml \
    --decision-log llm_tsc_decisions.json --raw-response-log llm_tsc_raw_responses.json --gui
@LastEditTime: 2026-06-13 22:33:30
'''
import re
import json
import argparse

from typing import Dict
from time import perf_counter

from openai import OpenAI
import httpx
from common.logger import logger

from methods.llm_tsc.memory import EpisodeMemory
from methods.llm_tsc.metrics import write_episode_metrics

from methods.llm_tsc.agent import SimplifiedReActAgent
from common.config import load_config
from methods.llm_tsc.prompts import PromptManager
from methods.llm_tsc.tools_adapter import TSCToolsAdapter
from methods.llm_tsc.tools_registry import ToolRegistry
from common.runtime import (
    add_simulation_arguments, append_decision_log, build_env as build_method_env,
    parse_phase_id,
)


def parse_agent_response(response: str) -> Dict:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    data = {}
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        for match in reversed(list(re.finditer(r"\{", response))):
            try:
                candidate, _ = json.JSONDecoder().raw_decode(response[match.start():])
                if isinstance(candidate, dict) and "decision" in candidate:
                    data = candidate
                    break
            except json.JSONDecodeError:
                continue

    if not isinstance(data, dict):
        data = {}

    decision = data.get("decision")
    phase_id = None
    if isinstance(decision, int) and not isinstance(decision, bool) and decision >= 0:
        phase_id = decision
    elif isinstance(decision, str):
        match = re.fullmatch(r"(?:Phase[- ]?)?(\d+)", decision.strip())
        if match:
            phase_id = int(match.group(1))

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


def build_env(args, num_seconds: int):
    return build_method_env(args, num_seconds, "llm_tsc")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal LLM-based TSC runner")
    add_simulation_arguments(parser)
    parser.add_argument("--raw-response-log", default=None, help="Optional JSONL path for raw LLM responses")
    parser.add_argument("--episode-summary", default=None, help="Episode summary JSON; also writes a steps CSV")
    parser.add_argument("--fail-on-api-error", action="store_true", help="Stop if the model API fails")
    parser.add_argument("--max-tool-rounds", type=int, default=2,
                        help="Maximum tool-query rounds per decision (default: 2; 0: direct decisions)")
    args = parser.parse_args()
    if args.max_tool_rounds < 0:
        parser.error("--max-tool-rounds must be nonnegative")

    config = load_config(args.config)
    if not config.validate():
        return

    client = OpenAI(
        api_key=config["OPENAI_API_KEY"],
        base_url=config["OPENAI_API_BASE"],
        timeout=60.0,
        http_client=httpx.Client(proxy=config["OPENAI_PROXY"], trust_env=False),
    )

    total_time = args.num_seconds or config["TOTAL_SIMULATION_TIME"]
    episode_started = perf_counter()
    step_records = []
    env = build_env(args, num_seconds=total_time)
    try:
        env.reset() # reset first so static intersection info is available for the prompt
        memory = EpisodeMemory()
        adapter = TSCToolsAdapter(env, memory)
        adapter.reset_episode()
        tools = ToolRegistry().register_from_object(adapter)
        agent = SimplifiedReActAgent(
            llm_client=client,
            tools_dict=tools,
            system_prompt=PromptManager.get_system_prompt(memory.render_static_context()),
            max_iterations=config["MAX_ITERATIONS"],
            model=config["OPENAI_API_MODEL"],
            temperature=config["OPENAI_TEMPERATURE"],
            verbose=config["LLM_VERBOSE"],
            max_tool_rounds=args.max_tool_rounds,
            extra_body=config["OPENAI_EXTRA_BODY"],
        )

        done = False
        sim_time = 0
        phase_id = 0
        llm_decisions = 0
        traditional_fallbacks = 0

        while not done:
            step_started = perf_counter()
            llm_decisions += 1
            api_calls = []
            tool_calls = []
            snapshot = adapter.begin_decision(sim_time)
            message = PromptManager.get_agent_message(
                sim_step=sim_time,
                last_step_action=phase_id,
                snapshot=snapshot,
            )
            response = agent.run(message, reset_history=True)
            api_calls.extend(agent.last_run_stats["api_calls"])
            tool_calls.extend(agent.last_run_stats["tool_calls"])
            if args.fail_on_api_error and any("error" in call for call in api_calls):
                raise RuntimeError("Model API failed; episode stopped. See the API error above.")
            append_decision_log(
                args.raw_response_log,
                {
                    "sim_time": sim_time,
                    "response_type": "initial",
                    "raw_response": response,
                },
            )
            parsed = parse_agent_response(response)
            if not parsed["has_valid_decision"] or not 0 <= parsed["phase_id"] < args.phase_num:
                logger.warning("Failed to parse LLM phase decision; requesting JSON reformat")
                response = agent.run(PromptManager.get_error_handling_prompt(), reset_history=False, allow_tools=False)
                api_calls.extend(agent.last_run_stats["api_calls"])
                tool_calls.extend(agent.last_run_stats["tool_calls"])
                if args.fail_on_api_error and any("error" in call for call in api_calls):
                    raise RuntimeError("Model API failed during decision reformat; episode stopped.")
                append_decision_log(
                    args.raw_response_log,
                    {
                        "sim_time": sim_time,
                        "response_type": "reformat",
                        "raw_response": response,
                    },
                )
                parsed = parse_agent_response(response)

            if parsed["has_valid_decision"] and 0 <= parsed["phase_id"] < args.phase_num:
                phase_id = parsed["phase_id"]
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
            env_state = {
                "traditional_decision": adapter.get_phase_recommendation()["recommendation"],
                "junction_situation": snapshot["situation"],
                "traffic_snapshot": snapshot,
            }
            mp_phase_id = parse_phase_id(env_state["traditional_decision"].get("recommended_phase"))
            consulted_mp = any(call["name"] == "get_phase_recommendation" and call["success"]
                               for call in tool_calls)
            record = {
                    "step": len(step_records),
                    "sim_time": sim_time,
                    "controller": "llm_tsc",
                    "source": decision_source,
                    "phase_id": phase_id,
                    "explanation": explanation,
                    "mp_phase_id": mp_phase_id,
                    "consulted_mp": consulted_mp,
                    "expert_advice_provided": 'expert_advice' in snapshot,
                    "matches_mp": phase_id == mp_phase_id,
                    "api_call_count": len(api_calls),
                    "api_error_count": sum("error" in call for call in api_calls),
                    "tool_call_count": len(tool_calls),
                    "api_calls": api_calls,
                    "tool_calls": tool_calls,
                    "llm_latency_ms": sum(call["latency_ms"] for call in api_calls),
                    "tool_latency_ms": sum(call["latency_ms"] for call in tool_calls),
                    **env_state,
            }
            logger.info(
                "Decision at {}s: source={}, phase={}, traditional={}",
                sim_time,
                decision_source,
                phase_id,
                env_state.get("traditional_decision", {}).get("recommended_phase"),
            )

            memory.remember(snapshot, phase_id)
            record["decision_latency_ms"] = (perf_counter() - step_started) * 1000
            env_started = perf_counter()
            _, _, _, done, info = env.step(action=phase_id)
            record["env_step_latency_ms"] = (perf_counter() - env_started) * 1000
            sim_time = info["step_time"]
            record["next_sim_time"] = sim_time
            record["step_latency_ms"] = (perf_counter() - step_started) * 1000
            step_records.append(record)
            append_decision_log(args.decision_log, record)
            logger.info("Step {}: sim={}s, latency={:.2f}s, tools={}, consulted_mp={}",
                        record["step"], record["sim_time"], record["step_latency_ms"] / 1000,
                        record["tool_call_count"], record["consulted_mp"])

            if sim_time >= total_time:
                break

    finally:
        env.close()
    if args.episode_summary:
        write_episode_metrics(step_records, args.episode_summary, {
            "scenario": args.scenario, "event_config": args.event_config,
            "model": config["OPENAI_API_MODEL"], "config_file": config.yaml_path,
            "model_extra_body": config["OPENAI_EXTRA_BODY"], "requested_simulation_seconds": total_time,
            "final_sim_time": sim_time, "episode_wall_seconds": perf_counter() - episode_started,
            "decision_log": args.decision_log,
            "max_tool_rounds": args.max_tool_rounds,
            "tool_names": list(tools),
            "system_prompt_chars": len(agent.system_prompt),
        })
    logger.info(
        "Finished simulation with {} LLM decisions and {} traditional fallbacks",
        llm_decisions,
        traditional_fallbacks,
    )


if __name__ == "__main__":
    main()
