import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2024, 12, 31)
DATE_RANGE = pd.date_range(START_DATE, END_DATE, freq="D")

PRODUCTS   = [f"SKU-{str(i).zfill(3)}" for i in range(1, 21)]
WAREHOUSES = ["WH-NORTH", "WH-SOUTH", "WH-EAST", "WH-WEST"]
SUPPLIERS  = [f"SUP-{str(i).zfill(2)}" for i in range(1, 9)]
REGIONS    = ["North", "South", "East", "West"]
OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_noise(series, pct=0.05):
    noise = np.random.normal(0, pct * series.mean(), size=len(series))
    return np.clip(series + noise, 0, None)

def seasonal_demand(dates, base, amplitude=0.3):
    t = np.arange(len(dates))
    seasonal = base * (1 + amplitude * np.sin(2 * np.pi * t / 365))
    trended  = seasonal * (1 + 0.0002 * t)
    return add_noise(trended, pct=0.08).astype(int)

def generate_demand():
    rows = []
    for sku in PRODUCTS:
        base = np.random.randint(50, 300)
        amp  = np.random.uniform(0.1, 0.5)
        for wh in WAREHOUSES:
            demand = seasonal_demand(DATE_RANGE, base, amp)
            for i, date in enumerate(DATE_RANGE):
                rows.append({"date": date.date(), "sku": sku, "warehouse": wh,
                             "demand": max(0, demand[i]), "region": REGIONS[WAREHOUSES.index(wh)]})
    df = pd.DataFrame(rows)
    stockout_idx = df.sample(frac=0.02).index
    df.loc[stockout_idx, "stockout"] = 1
    df["stockout"] = df["stockout"].fillna(0).astype(int)
    df.to_csv(f"{OUTPUT_DIR}/demand_history.csv", index=False)
    print(f"  demand_history.csv      — {len(df):,} rows")
    return df

def generate_inventory(demand_df):
    rows = []
    for sku in PRODUCTS:
        for wh in WAREHOUSES:
            stock = np.random.randint(500, 2000)
            rop   = np.random.randint(100, 400)
            for date in pd.date_range(START_DATE, END_DATE, freq="W"):
                wd = demand_df[(demand_df["sku"]==sku)&(demand_df["warehouse"]==wh)&
                               (demand_df["date"]>=(date-timedelta(7)).date())&
                               (demand_df["date"]<date.date())]["demand"].sum()
                stock = max(0, stock - wd)
                if stock < rop: stock += np.random.randint(500, 1500)
                rows.append({"date": date.date(), "sku": sku, "warehouse": wh,
                             "stock_level": int(stock), "reorder_point": rop,
                             "below_reorder": int(stock < rop)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/inventory_snapshots.csv", index=False)
    print(f"  inventory_snapshots.csv — {len(df):,} rows")
    return df

def generate_suppliers():
    rows = []
    for sup in SUPPLIERS:
        for sku in random.sample(PRODUCTS, k=random.randint(5, 15)):
            rows.append({"supplier": sup, "sku": sku,
                         "unit_cost": round(np.random.uniform(5, 80), 2),
                         "avg_lead_time_days": np.random.randint(3, 21),
                         "reliability_score": round(np.random.uniform(0.7, 1.0), 3),
                         "min_order_qty": np.random.choice([50, 100, 200, 500]),
                         "country": np.random.choice(["US","CN","DE","IN","MX"])})
    df = pd.DataFrame(rows).drop_duplicates(subset=["supplier","sku"])
    df.to_csv(f"{OUTPUT_DIR}/suppliers.csv", index=False)
    print(f"  suppliers.csv           — {len(df):,} rows")
    return df

def generate_warehouses():
    df = pd.DataFrame({
        "warehouse": WAREHOUSES,
        "city": ["Chicago","Atlanta","New York","Los Angeles"],
        "lat":  [41.8781, 33.7490, 40.7128, 34.0522],
        "lon":  [-87.6298,-84.3880,-74.0060,-118.2437],
        "capacity": [10000, 8000, 12000, 9500],
        "monthly_fixed_cost": [45000, 38000, 55000, 42000],
    })
    df.to_csv(f"{OUTPUT_DIR}/warehouses.csv", index=False)
    print(f"  warehouses.csv          — {len(df):,} rows")
    return df

def generate_shipping_lanes():
    rows = [{"origin": s, "destination": d,
             "distance_km": np.random.randint(300, 3000),
             "cost_per_unit": round(np.random.uniform(0.5, 5.0), 2),
             "transit_days": np.random.randint(1, 7),
             "carrier": np.random.choice(["FedEx","UPS","USPS","DHL"])}
            for s in WAREHOUSES for d in WAREHOUSES if s != d]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/shipping_lanes.csv", index=False)
    print(f"  shipping_lanes.csv      — {len(df):,} rows")
    return df

if __name__ == "__main__":
    print("\nGenerating synthetic supply chain data...\n")
    demand_df = generate_demand()
    generate_inventory(demand_df)
    generate_suppliers()
    generate_warehouses()
    generate_shipping_lanes()
    print("\nDone! Check data/raw/ folder\n")