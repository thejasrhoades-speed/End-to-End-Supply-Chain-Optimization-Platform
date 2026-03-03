import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import numpy as np

# Page Configuration
st.set_page_config(page_title="AI Supply Chain Optimizer", layout="wide")

st.title("📦 AI Supply Chain Optimization Platform")
st.markdown("### Real-time Inventory Insights & Demand Forecasting")

# 1. Sidebar for "What-If" Simulation
st.sidebar.header("Simulation Settings")
lead_time_buffer = st.sidebar.slider("Additional Lead Time (Days)", 0, 14, 0)

# 2. API Connection
API_URL = "http://127.0.0.1:8000/inventory/targets"

try:
    response = requests.get(API_URL)
    df = pd.DataFrame(response.json())

    # 3. Dynamic Simulation Logic
    # Re-calculating safety stock based on slider input
    z_score = 1.96  # 95% Confidence
    base_lt = 7     # Original Lead Time
    total_lt = base_lt + lead_time_buffer
    
    if lead_time_buffer > 0:
        # Update reorder point: (Daily Avg * New LT) + (Safety Stock for New LT)
        df['reorder_point'] = (df['daily_avg'] * total_lt) + \
                              (z_score * df['demand_std'] * np.sqrt(total_lt))
        st.sidebar.warning(f"⚠️ Simulation Active: {total_lt} Day Lead Time")
    else:
        st.sidebar.success("✅ Standard Lead Time (7 Days)")

    # 4. Executive Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total SKUs Tracked", len(df))
    col2.metric("Avg. Optimal Order Qty", int(df['optimal_order_qty'].mean()))
    # Critical items are those where reorder point exceeds a safety threshold
    col3.metric("Critical Reorder Points", len(df[df['reorder_point'] > 1200]))

    # 5. Inventory Strategy Visualization
    st.subheader("Inventory Strategy by SKU")
    fig = px.scatter(
        df, x="optimal_order_qty", y="reorder_point", 
        hover_name="sku", size="annual_demand", color="annual_demand",
        labels={"optimal_order_qty": "Order Quantity (EOQ)", "reorder_point": "Reorder Threshold"},
        title="Optimal Order Quantity vs. Reorder Threshold",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    # 2026 Update: using width='stretch' instead of use_container_width
    st.plotly_chart(fig, width='stretch')

    # 6. Detailed Data View
    st.subheader("Detailed Optimization Targets")
    st.dataframe(df.style.highlight_max(axis=0, subset=['annual_demand']), width='stretch')

except Exception as e:
    st.error(f"🔴 Connection Error: Is your FastAPI server running? (Error: {e})")