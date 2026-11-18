
import csv
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT_DIR / "reports"
MONITORING_FILE = REPORTS_DIR / "api_monitoring.csv"

FIELDNAMES = [
    "timestamp",
    "status_code",
    "response_time_ms",
    "cache_hit",
    "recommendation_count",
]


def record_api_metrics(
    status_code,
    response_time_ms,
    cache_hit,
    recommendation_count,
):
    """Append one API request's performance metrics to a CSV file."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = MONITORING_FILE.exists()

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status_code": status_code,
        "response_time_ms": round(response_time_ms, 2),
        "cache_hit": cache_hit,
        "recommendation_count": recommendation_count,
    }

    with MONITORING_FILE.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)
