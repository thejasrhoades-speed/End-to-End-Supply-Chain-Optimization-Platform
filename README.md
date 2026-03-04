## 🎥 Dashboard Preview

![AI Supply Chain Optimization Dashboard](docs/images/dashboard_preview.png)

> **Real-time Inventory Insights & Demand Forecasting** — Interactive simulation with adjustable lead times, EOQ scatter plots, and detailed per-SKU optimization targets.

---

## 📊 Performance Metrics

### Inventory Optimization
| Metric | Value |
|--------|-------|
| Total SKUs Tracked | 20 |
| Avg. Optimal Order Quantity (EOQ) | 6,044 units |
| Critical Reorder Points Identified | 15 / 20 SKUs |
| Annual Demand Range | 510K – 735K units/SKU |
| Safety Stock Coverage | Calculated per SKU (std-dev adjusted) |

### Demand Forecasting
| Metric | Value |
|--------|-------|
| Forecast Horizon | 30-day rolling window |
| Model | Time-series ML (per-SKU) |
| Lead Time Simulation | Dynamic (1–30 day adjustment) |
| Reorder Point Accuracy | 95%+ service level target |

### System Highlights
- ⚡ **Real-time simulation** — adjust lead time with a slider and watch all metrics update instantly  
- 📦 **20 SKUs** tracked across 4 warehouses  
- 🧮 **EOQ model** with demand variability and safety stock calculations  
- 📈 **Scatter plot visualization** — EOQ vs. Reorder Point, color-coded by annual demand  
- 🗂️ **Detailed optimization table** — daily avg, demand std dev, safety stock, reorder point per SKU  

---

## 🖥️ Dashboard Features

### Simulation Settings (Left Panel)
- Adjustable **Additional Lead Time** slider (1–30 days)
- Live simulation mode indicator with current settings
- All charts and tables update in real-time

### Inventory Strategy View
- **Scatter plot**: Optimal Order Quantity (EOQ) vs. Reorder Threshold, bubble size and color = annual demand
- Identifies which SKUs require the most critical attention (15 out of 20 flagged)

### Detailed Optimization Targets Table
Displays per-SKU breakdown:
- `annual_demand` — total yearly units  
- `demand_std` — demand variability  
- `daily_avg` — average daily demand  
- `optimal_order_qty` — calculated EOQ  
- `safety_stock` — buffer stock  
- `reorder_point` — trigger threshold  

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/thejasrhoades-speed/End-to-End-Supply-Chain-Optimization-Platform.git
cd End-to-End-Supply-Chain-Optimization-Platform

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python generate_data.py

# Run the dashboard
streamlit run dashboard/app.py

# Run the API (separate terminal)
uvicorn main:app --reload
```

Open your browser at `http://localhost:8501` for the dashboard and `http://localhost:8000/docs` for the API.

---

## 🔗 Live Demo

> 🚧 *Deployment in progress — demo links coming soon*  
> **Dashboard**: [Streamlit Cloud](#) | **API Docs**: [Railway](#)
