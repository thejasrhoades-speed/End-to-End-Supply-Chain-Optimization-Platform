import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="AI Supply Chain Optimizer", layout="wide")

st.title("📦 AI Supply Chain Optimization Platform")
st.markdown("### Real-time Inventory Insights & Demand Forecasting")

# 1. Sidebar for "What-If" Simulation
st.sidebar.header("Simulation Settings")
lead_time_buffer = st.sidebar.slider("Additional Lead Time (Days)", 0, 14, 0)

# 2. Generate data directly (no API needed)
@st.cache_data
def generate_inventory_data():
    np.random.seed(42)
    skus = [f"SKU-{str(i).zfill(3)}" for i in range(1, 21)]
    data = []
    for sku in skus:
        annual_demand = np.random.randint(400000, 800000)
        demand_std = np.random.uniform(20, 55)
        daily_avg = annual_demand / 365
        unit_cost = np.random.uniform(20, 100)
        holding_cost = unit_cost * 0.25
        ordering_cost = 150

        # EOQ
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)

        # Safety stock (95% service level)
        z = 1.96
        safety_stock = z * demand_std * np.sqrt(7)

        # Reorder point
        reorder_point = (daily_avg * 7) + safety_stock

        data.append({
            "sku": sku,
            "annual_demand": int(annual_demand),
            "demand_std": round(demand_std, 2),
            "daily_avg": round(daily_avg, 2),
            "optimal_order_qty": round(eoq, 2),
            "safety_stock": round(safety_stock, 2),
            "reorder_point": round(reorder_point, 2),
            "unit_cost": round(unit_cost, 2),
        })
    return pd.DataFrame(data)

df = generate_inventory_data()

# 3. Dynamic Simulation Logic
z_score = 1.96
base_lt = 7
total_lt = base_lt + lead_time_buffer

if lead_time_buffer > 0:
    df = df.copy()
    df["safety_stock"] = z_score * df["demand_std"] * np.sqrt(total_lt)
    df["reorder_point"] = (df["daily_avg"] * total_lt) + df["safety_stock"]
    st.sidebar.warning(f"⚠️ Simulation Active: {total_lt} Day Lead Time")

# 4. KPI Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total SKUs Tracked", len(df))
col2.metric("Avg. Optimal Order Qty", int(df["optimal_order_qty"].mean()))
col3.metric("Critical Reorder Points", len(df[df["reorder_point"] > 1200]))

# 5. Inventory Strategy Visualization
st.subheader("Inventory Strategy by SKU")
fig = px.scatter(
    df,
    x="optimal_order_qty",
    y="reorder_point",
    color="annual_demand",
    hover_name="sku",
    size="annual_demand",
    labels={"optimal_order_qty": "Order Quantity (EOQ)", "reorder_point": "Reorder Threshold"},
    title="Optimal Order Quantity vs. Reorder Threshold",
    color_continuous_scale="viridis",
    template="plotly_dark",
)
st.plotly_chart(fig, use_container_width=True)

# 6. Detailed Table
st.subheader("Detailed Optimization Targets")
st.dataframe(df.style.format({
    "annual_demand": "{:,.0f}",
    "demand_std": "{:.2f}",
    "daily_avg": "{:.2f}",
    "optimal_order_qty": "{:,.2f}",
    "safety_stock": "{:.2f}",
    "reorder_point": "{:,.2f}",
    "unit_cost": "${:.2f}",
}), use_container_width=True)

# 7. Demand Forecast Section
st.subheader("📈 30-Day Demand Forecast")
selected_sku = st.selectbox("Select SKU", df["sku"].tolist())
sku_row = df[df["sku"] == selected_sku].iloc[0]

dates = pd.date_range(start=pd.Timestamp.now(), periods=30, freq="D")
base = sku_row["daily_avg"]
forecast = base + np.cumsum(np.random.normal(0, sku_row["demand_std"] * 0.1, 30))
lower = forecast * 0.88
upper = forecast * 1.12

forecast_df = pd.DataFrame({
    "Date": dates,
    "Forecast": forecast,
    "Lower Bound": lower,
    "Upper Bound": upper,
})

fig2 = px.line(forecast_df, x="Date", y=["Forecast", "Lower Bound", "Upper Bound"],
               title=f"Demand Forecast - {selected_sku}",
               template="plotly_dark",
               color_discrete_map={"Forecast": "#FF6B35", "Lower Bound": "#888", "Upper Bound": "#888"})
st.plotly_chart(fig2, use_container_width=True)

st.caption("Built with ❤️ using Python, Streamlit, and Plotly")