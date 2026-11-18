
from pathlib import Path
import json

import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

REPORTS.mkdir(exist_ok=True)


# Load datasets
customers = pd.read_csv(RAW / "customers.csv")
products = pd.read_csv(RAW / "products.csv")
interactions = pd.read_csv(RAW / "interactions.csv")

# Parse timestamps
interactions["timestamp"] = pd.to_datetime(
    interactions["timestamp"],
    errors="coerce"
)


# --------------------------------------------------
# DATA QUALITY CHECKS
# --------------------------------------------------

checks = {
    "customers": {
        "rows": len(customers),
        "columns": len(customers.columns),
        "missing_values": int(customers.isnull().sum().sum()),
        "duplicate_rows": int(customers.duplicated().sum()),
        "duplicate_user_ids": int(customers["user_id"].duplicated().sum()),
    },
    "products": {
        "rows": len(products),
        "columns": len(products.columns),
        "missing_values": int(products.isnull().sum().sum()),
        "duplicate_rows": int(products.duplicated().sum()),
        "duplicate_item_ids": int(products["item_id"].duplicated().sum()),
    },
    "interactions": {
        "rows": len(interactions),
        "columns": len(interactions.columns),
        "missing_values": int(interactions.isnull().sum().sum()),
        "duplicate_rows": int(interactions.duplicated().sum()),
        "invalid_timestamps": int(interactions["timestamp"].isnull().sum()),
    },
}


# --------------------------------------------------
# RELATIONSHIP VALIDATION
# --------------------------------------------------

valid_users = set(customers["user_id"])
valid_items = set(products["item_id"])

unknown_users = int(
    (~interactions["user_id"].isin(valid_users)).sum()
)

unknown_items = int(
    (~interactions["item_id"].isin(valid_items)).sum()
)

valid_events = {"view", "click", "save", "enroll"}

invalid_events = int(
    (~interactions["event_type"].isin(valid_events)).sum()
)


# --------------------------------------------------
# BUSINESS ANALYTICS
# --------------------------------------------------

event_distribution = {
    str(k): int(v)
    for k, v in interactions["event_type"].value_counts().items()
}

category_distribution = {
    str(k): int(v)
    for k, v in products["category"].value_counts().items()
}

consent_distribution = {
    str(k): int(v)
    for k, v in customers["consent"].value_counts().items()
}

availability_distribution = {
    str(k): int(v)
    for k, v in products["available"].value_counts().items()
}

unique_interaction_pairs = int(
    interactions[["user_id", "item_id"]].drop_duplicates().shape[0]
)

possible_pairs = len(customers) * len(products)

interaction_density = (
    unique_interaction_pairs / possible_pairs * 100
)


# --------------------------------------------------
# VALIDATION SUMMARY
# --------------------------------------------------

validation = {
    "unknown_user_references": unknown_users,
    "unknown_product_references": unknown_items,
    "invalid_event_types": invalid_events,
    "unique_interaction_pairs": unique_interaction_pairs,
    "possible_user_item_pairs": possible_pairs,
    "interaction_density_percent": round(interaction_density, 2),
}

all_checks_passed = all([
    all(v.get("missing_values", 0) == 0 for v in checks.values()),
    all(v.get("duplicate_rows", 0) == 0 for v in checks.values()),
    checks["customers"]["duplicate_user_ids"] == 0,
    checks["products"]["duplicate_item_ids"] == 0,
    checks["interactions"]["invalid_timestamps"] == 0,
    unknown_users == 0,
    unknown_items == 0,
    invalid_events == 0,
])


# --------------------------------------------------
# SAVE REPORT
# --------------------------------------------------

report = {
    "project": "Personalized Financial Product Recommendation Engine",
    "dataset_type": "Synthetic",
    "quality_status": "PASS" if all_checks_passed else "FAIL",
    "dataset_quality": checks,
    "relationship_validation": validation,
    "business_summary": {
        "event_distribution": event_distribution,
        "category_distribution": category_distribution,
        "consent_distribution": consent_distribution,
        "availability_distribution": availability_distribution,
    },
}

report_path = REPORTS / "data_quality_report.json"

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)


# --------------------------------------------------
# CONSOLE OUTPUT
# --------------------------------------------------

print("\n" + "=" * 55)
print("DATA QUALITY & EDA REPORT")
print("=" * 55)

print("\nDATASET QUALITY")

for name, values in checks.items():
    print(f"\n{name.upper()}")

    for metric, value in values.items():
        print(f"{metric}: {value:,}")

print("\nRELATIONSHIP VALIDATION")

for key, value in validation.items():
    print(f"{key}: {value}")

print("\nEVENT DISTRIBUTION")
print(interactions["event_type"].value_counts())

print("\nPRODUCT CATEGORY DISTRIBUTION")
print(products["category"].value_counts())

print("\nCUSTOMER CONSENT")
print(customers["consent"].value_counts())

print("\nFINAL STATUS:", report["quality_status"])
print("Report saved:", report_path)
