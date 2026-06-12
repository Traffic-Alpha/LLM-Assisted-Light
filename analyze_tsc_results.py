'''
@Author: WANG Maonan
@Date: 2026-06-12 23:52:00
@Description: Compare TSC tripinfo results with regular/special vehicle metrics.
'''
import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


SPECIAL_KEYWORDS = ("rescue", "ambulance", "police", "fire")
NUMERIC_FIELDS = ("duration", "waitingTime", "timeLoss", "routeLength")


def is_special_vehicle(attrs: Dict[str, str]) -> bool:
    text = f"{attrs.get('id', '')} {attrs.get('vType', '')}".lower()
    return any(keyword in text for keyword in SPECIAL_KEYWORDS)


def is_completed(attrs: Dict[str, str]) -> bool:
    return attrs.get("arrival", "-1") != "-1.00" and attrs.get("vaporized", "") == ""


def summarize_group(rows: Iterable[Dict[str, str]]) -> Dict[str, Any]:
    rows = list(rows)
    completed = [row for row in rows if is_completed(row)]
    summary: Dict[str, Any] = {
        "vehicles": len(rows),
        "completed": len(completed),
        "completion_rate": round(len(completed) / len(rows), 4) if rows else 0.0,
    }

    for field in NUMERIC_FIELDS:
        values = [float(row[field]) for row in rows if field in row]
        summary[field] = {
            "avg": round(sum(values) / len(values), 3) if values else 0.0,
            "sum": round(sum(values), 3) if values else 0.0,
            "max": round(max(values), 3) if values else 0.0,
        }

    return summary


def read_tripinfo(path: Path) -> List[Dict[str, str]]:
    root = ET.parse(path).getroot()
    return [dict(node.attrib) for node in root.findall("tripinfo")]


def summarize_tripinfo(path: Path) -> Dict[str, Any]:
    rows = read_tripinfo(path)
    special = [row for row in rows if is_special_vehicle(row)]
    regular = [row for row in rows if not is_special_vehicle(row)]
    return {
        "tripinfo": str(path),
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
    args = parser.parse_args()

    output = {}
    for result in args.results:
        name, path = parse_result_arg(result)
        output[name] = summarize_tripinfo(path)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
