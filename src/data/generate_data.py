
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"

RANDOM_SEED = 42

N_CUSTOMERS = 1_000
N_PRODUCTS = 50
INTERACTIONS_PER_USER = 20

rng = np.random.default_rng(RANDOM_SEED)


# --------------------------------------------------
# 1. GENERATE CUSTOMERS
# --------------------------------------------------

categories = [
    "savings",
    "budgeting",
    "credit",
    "investing",
    "insurance",
    "financial_planning",
]

segments = ["student", "young_professional", "family", "experienced"]

customers = []

for i in range(N_CUSTOMERS):

    interests = rng.choice(
        categories,
        size=int(rng.integers(1, 4)),
        replace=False,
    ).tolist()

    customers.append({
        "user_id": f"U{i + 1:05d}",
        "age_group": rng.choice(
            ["18-24", "25-34", "35-44", "45-54", "55+"]
        ),
        "customer_segment": rng.choice(segments),
        "interests": "|".join(interests),
        "consent": bool(rng.choice([True, False], p=[0.85, 0.15])),
    })

customers_df = pd.DataFrame(customers)


# --------------------------------------------------
# 2. GENERATE FINANCIAL PRODUCTS
# --------------------------------------------------

product_templates = {
    "savings": [
        "Emergency Fund Guide",
        "Savings Fundamentals",
        "Understanding Interest",
    ],
    "budgeting": [
        "Monthly Budget Planner",
        "Expense Tracking Guide",
        "Personal Budgeting Basics",
    ],
    "credit": [
        "Credit Score Education",
        "Understanding Credit Reports",
        "Responsible Credit Management",
    ],
    "investing": [
        "Investment Fundamentals",
        "Understanding Diversification",
        "Long-Term Investing Education",
    ],
    "insurance": [
        "Insurance Basics",
        "Understanding Life Insurance",
        "Risk Protection Education",
    ],
    "financial_planning": [
        "Financial Planning Guide",
        "Retirement Planning Basics",
        "Financial Goal Setting",
    ],
}

products = []

for i in range(N_PRODUCTS):

    category = categories[i % len(categories)]

    products.append({
        "item_id": f"P{i + 1:03d}",
        "product_name": rng.choice(product_templates[category]),
        "category": category,
        "description": (
            f"Educational resource about {category.replace('_', ' ')} "
            "and personal financial knowledge."
        ),
        "tags": f"{category}|financial_education|personal_finance",
        "available": bool(rng.choice([True, False], p=[0.90, 0.10])),
        "minimum_age": 18,
    })

products_df = pd.DataFrame(products)


# --------------------------------------------------
# 3. GENERATE USER-PRODUCT INTERACTIONS
# --------------------------------------------------

event_types = ["view", "click", "save", "enroll"]
event_probabilities = [0.55, 0.25, 0.13, 0.07]

interaction_records = []

product_categories = products_df["category"].to_numpy()

for _, customer in customers_df.iterrows():

    user_interests = customer["interests"].split("|")

    # Products matching customer interests receive
    # higher selection probabilities.
    weights = np.array([
        5.0 if category in user_interests else 1.0
        for category in product_categories
    ])

    probabilities = weights / weights.sum()

    selected_indices = rng.choice(
        N_PRODUCTS,
        size=INTERACTIONS_PER_USER,
        replace=True,
        p=probabilities,
    )

    for product_index in selected_indices:

        event = rng.choice(
            event_types,
            p=event_probabilities,
        )

        interaction_records.append({
            "user_id": customer["user_id"],
            "item_id": products_df.iloc[product_index]["item_id"],
            "event_type": event,
            "timestamp": (
                pd.Timestamp("2026-01-01")
                + pd.Timedelta(
                    days=int(rng.integers(0, 180)),
                    seconds=int(rng.integers(0, 86400)),
                )
            ),
        })

interactions_df = pd.DataFrame(interaction_records)

interactions_df = interactions_df.sort_values(
    "timestamp"
).reset_index(drop=True)


# --------------------------------------------------
# 4. SAVE DATASETS
# --------------------------------------------------

RAW_DIR.mkdir(parents=True, exist_ok=True)

customers_df.to_csv(
    RAW_DIR / "customers.csv",
    index=False,
)

products_df.to_csv(
    RAW_DIR / "products.csv",
    index=False,
)

interactions_df.to_csv(
    RAW_DIR / "interactions.csv",
    index=False,
)


# --------------------------------------------------
# 5. DATASET SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("FINANCIAL RECOMMENDATION DATA GENERATOR")
print("=" * 55)

print(f"\nCustomers:    {len(customers_df):,}")
print(f"Products:     {len(products_df):,}")
print(f"Interactions: {len(interactions_df):,}")

print("\nEvent Distribution:")
print(interactions_df["event_type"].value_counts())

print("\nProduct Categories:")
print(products_df["category"].value_counts())

print("\nConsent Distribution:")
print(customers_df["consent"].value_counts())

print("\nMissing Values:")
print("Customers:", customers_df.isnull().sum().sum())
print("Products:", products_df.isnull().sum().sum())
print("Interactions:", interactions_df.isnull().sum().sum())

print("\nSaved files:")
for file in RAW_DIR.glob("*.csv"):
    print(f" - {file.name}")

print("\nDATA GENERATION COMPLETED SUCCESSFULLY!")
