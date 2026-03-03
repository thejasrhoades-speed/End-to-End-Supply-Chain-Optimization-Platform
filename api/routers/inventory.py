from fastapi import APIRouter, HTTPException
import pandas as pd
import os

router = APIRouter(prefix="/inventory", tags=["Inventory Optimization"])

# The path to the 'Gold' you just generated
DATA_PATH = 'data/processed/inventory_targets.csv'

@router.get("/targets")
def get_all_targets():
    """Fetch all calculated SKU targets (EOQ, Safety Stock, Reorder Point)."""
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Optimization data not found.")
    
    df = pd.read_csv(DATA_PATH)
    return df.to_dict(orient='records')

@router.get("/alerts")
def get_stock_alerts():
    """Items requiring immediate attention (Highest Reorder Points)."""
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Data source missing.")
    
    df = pd.read_csv(DATA_PATH)
    # Filter for top 5 critical reorder items
    alerts = df.sort_values(by='reorder_point', ascending=False).head(5)
    return alerts.to_dict(orient='records')