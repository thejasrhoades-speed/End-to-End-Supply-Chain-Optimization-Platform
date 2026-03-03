import pandas as pd
import numpy as np
import os

def calculate_inventory_targets():
    print("🎯 Phase 3: Running Inventory Optimization...")
    
    # Path to the data we generated in Phase 1
    data_path = 'data/raw/demand_history.csv'
    if not os.path.exists(data_path):
        print("❌ Error: demand_history.csv not found!")
        return

    df = pd.read_csv(data_path)
    
    # Business Logic Constants
    cost_to_order = 50.0   # Cost per replenishment order
    holding_cost = 2.0     # Annual cost to store one unit
    service_level_z = 1.96 # 95% confidence level to prevent stockouts

    # Calculate stats per SKU (Note: matching your 'sku' column name)
    stats = df.groupby('sku')['demand'].agg(['sum', 'std', 'mean']).reset_index()
    stats.columns = ['sku', 'annual_demand', 'demand_std', 'daily_avg']

    # EOQ Formula: sqrt((2 * Demand * Order_Cost) / Holding_Cost)
    stats['optimal_order_qty'] = np.sqrt((2 * stats['annual_demand'] * cost_to_order) / holding_cost)

    # Safety Stock & Reorder Point (Assuming 7-day lead time)
    lead_time = 7
    stats['safety_stock'] = service_level_z * stats['demand_std'] * np.sqrt(lead_time)
    stats['reorder_point'] = (stats['daily_avg'] * lead_time) + stats['safety_stock']

    # Save to the 'processed' folder
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    stats.to_csv(f'{output_dir}/inventory_targets.csv', index=False)
    
    print(f"✅ Success! Targets saved to {output_dir}/inventory_targets.csv")

if __name__ == "__main__":
    calculate_inventory_targets()