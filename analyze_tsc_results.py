'''
@Author: WANG Maonan
@Date: 2026-06-12 23:52:00
@Description: Compare TSC tripinfo results with regular/special vehicle metrics.
'''
import argparse
import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List


SPECIAL_KEYWORDS = ("rescue", "ambulance", "police", "fire")
NUMERIC_FIELDS = ("duration", "waitingTime", "timeLoss", "routeLength")


def is_special_vehicle(attrs: Dict[str, str]) -> bool:
    text = f"{attrs.get('id', '')} {attrs.get('vType', '')}".lower()
    return any(keyword in text for keyword in SPECIAL_KEYWORDS)


def is_completed(attrs: Dict[str, str]) -> bool:
    return float(attrs.get("arrival", -1)) >= 0 and attrs.get("vaporized", "") == ""


def summarize_group(rows: Iterable[Dict[str, str]]) -> Dict[str, Any]:
    rows = list(rows)
    completed = [row for row in rows if is_completed(row)]
    summary: Dict[str, Any] = {
        "vehicles": len(rows),
        "completed": len(completed),
        "unfinished": len(rows) - len(completed),
        "completion_rate": round(len(completed) / len(rows), 4) if rows else 0.0,
    }

    for field in NUMERIC_FIELDS:
        values = [float(row[field]) for row in completed if field in row]
        summary[field] = {
            "avg": round(sum(values) / len(values), 3) if values else 0.0,
            "sum": round(sum(values), 3) if values else 0.0,
            "max": round(max(values), 3) if values else 0.0,
        }

    summary["travel_time"] = dict(summary["duration"])
    summary["observed_waiting_time_all_vehicles"] = {
        "avg": round(sum(float(row.get("waitingTime", 0)) for row in rows) / len(rows), 3) if rows else 0.0,
        "sum": round(sum(float(row.get("waitingTime", 0)) for row in rows), 3),
    }
    return summary


def read_tripinfo(path: Path) -> List[Dict[str, str]]:
    root = ET.parse(path).getroot()
    return [dict(node.attrib) for node in root.findall("tripinfo")]


def summarize_tripinfo(path: Path, obstacle_ids: set[str] | None = None) -> Dict[str, Any]:
    obstacle_ids = obstacle_ids or set()
    all_rows = read_tripinfo(path)
    excluded = [row for row in all_rows if row.get("id") in obstacle_ids
                or row.get("vType") in ("parked_car", "barrier")]
    excluded_ids = {row["id"] for row in excluded}
    rows = [row for row in all_rows if row["id"] not in excluded_ids]
    special = [row for row in rows if is_special_vehicle(row)]
    regular = [row for row in rows if not is_special_vehicle(row)]
    return {
        "tripinfo": str(path),
        "metric_definition": "duration/travel_time and waitingTime summarize completed trips only; observed_waiting_time_all_vehicles includes cumulative waiting of unfinished trips.",
        "excluded_obstacles": sorted(excluded_ids),
        "all": summarize_group(rows),
        "regular": summarize_group(regular),
        "special": summarize_group(special),
        "special_vehicles": [
            {
                "id": row.get("id"),
                "vType": row.get("vType"),
                "completed": is_completed(row),
                "depart": float(row.get("depart", 0)),
                "arrival": float(row.get("arrival", -1)),
                "duration": float(row.get("duration", 0)),
                "waitingTime": float(row.get("waitingTime", 0)),
                "timeLoss": float(row.get("timeLoss", 0)),
                "vaporized": row.get("vaporized", ""),
            }
            for row in special
        ],
    }


def parse_result_arg(value: str) -> tuple[str, Path]:
    if ":" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split(":", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TSC tripinfo results, including special vehicle efficiency."
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="Tripinfo result as [name:]path, e.g. llm:4way_llm_tsc.tripinfo.xml",
    )
    parser.add_argument("--event-config", default=None, help="Event YAML used to exclude accident obstacles")
    parser.add_argument("--output", default=None, help="Save comparison JSON and CSV")
    args = parser.parse_args()

    obstacle_ids = set()
    if args.event_config:
        from la_light.env.event_config import EventConfig
        obstacle_ids = {event.id for event in EventConfig(args.event_config).accidents}

    output = {}
    for result in args.results:
        name, path = parse_result_arg(result)
        output[name] = summarize_tripinfo(path, obstacle_ids)

    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")
        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
            columns = ["method", "vehicle_group", "vehicles", "completed", "unfinished",
                       "completion_rate", "travel_time_s", "waiting_time_s",
                       "observed_waiting_time_all_vehicles_s"]
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for name, result in output.items():
                for group in ("regular", "special"):
                    metrics = result[group]
                    writer.writerow({"method": name, "vehicle_group": group,
                        **{key: metrics[key] for key in ("vehicles", "completed", "unfinished", "completion_rate")},
                        "travel_time_s": metrics["travel_time"]["avg"],
                        "waiting_time_s": metrics["waitingTime"]["avg"],
                        "observed_waiting_time_all_vehicles_s": metrics["observed_waiting_time_all_vehicles"]["avg"],
                    })
    print(rendered)


if __name__ == "__main__":
    main()
