"""Small experiment outputs: per-decision CSV and episode summary."""
import csv
import json
from collections import Counter
from pathlib import Path


def latency_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {}
    def percentile(q):
        index = (len(ordered) - 1) * q
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        return round(ordered[low] + (ordered[high] - ordered[low]) * (index - low), 3)
    return {"mean": round(sum(ordered) / len(ordered), 3),
            "median": percentile(0.5), "p95": percentile(0.95),
            "min": round(ordered[0], 3), "max": round(ordered[-1], 3)}


def write_episode_metrics(records: list[dict], summary_path: str, metadata: dict) -> dict:
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    llm_records = [r for r in records if r["source"] == "llm"]
    consulted = [r for r in llm_records if r["consulted_mp"]]
    summary = {
        **metadata, "decision_steps": len(records),
        "llm_decisions": len(llm_records),
        "llm_consulted_mp_steps": len(consulted),
        "llm_without_mp_steps": sum(not r["consulted_mp"] for r in llm_records),
        "expert_advice_provided_steps": sum(r.get("expert_advice_provided", False) for r in records),
        "expert_advice_provided_and_matched_steps": sum(r.get("expert_advice_provided", False) and r["matches_mp"] for r in llm_records),
        "mp_fallback_steps": sum(r["source"] == "traditional_fallback" for r in records),
        "consulted_mp_and_matched_steps": sum(r["matches_mp"] for r in consulted),
        "consulted_mp_but_differed_steps": sum(not r["matches_mp"] for r in consulted),
        "llm_matches_mp_steps": sum(r["matches_mp"] for r in llm_records),
        "api_calls": sum(r["api_call_count"] for r in records),
        "api_errors": sum(r["api_error_count"] for r in records),
        "tool_calls": sum(r["tool_call_count"] for r in records),
        "tool_errors": sum(not call["success"] for r in records for call in r["tool_calls"]),
        "tool_errors_by_name": dict(Counter(call["name"] for r in records for call in r["tool_calls"] if not call["success"])),
        "tool_calls_by_name": dict(Counter(call["name"] for r in records for call in r["tool_calls"])),
        "latency_ms": {key: latency_summary([r[key] for r in records]) for key in
                       ("decision_latency_ms", "llm_latency_ms", "tool_latency_ms",
                        "env_step_latency_ms", "step_latency_ms")},
        "reference_definition": "consulted_mp counts successful model-requested get_phase_recommendation calls; expert_advice_provided records MP advice included in the supplied snapshot. Matching a phase alone does not prove use of the advice.",
    }
    summary["latency_ms"]["api_call_latency_ms"] = latency_summary(
        [call["latency_ms"] for record in records for call in record["api_calls"]])
    total_step_ms = sum(record["step_latency_ms"] for record in records)
    summary["api_wall_share_percent"] = round(
        100 * sum(record["llm_latency_ms"] for record in records) / total_step_ms, 3) if total_step_ms else 0
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    columns = ["step", "sim_time", "next_sim_time", "source", "phase_id", "mp_phase_id",
               "consulted_mp", "expert_advice_provided", "matches_mp", "api_call_count", "api_error_count",
               "tool_call_count", "tool_names", "decision_latency_ms", "llm_latency_ms",
               "tool_latency_ms", "env_step_latency_ms", "step_latency_ms"]
    with path.with_suffix(".steps.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, False) if key == 'expert_advice_provided' else record[key]
                   for key in columns if key != "tool_names"}
            row["tool_names"] = ";".join(c["name"] for c in record["tool_calls"])
            writer.writerow(row)
    return summary
