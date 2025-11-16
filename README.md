# Retail Analytics & Sales Forecasting  


---

## 📌 Overview  
A **full-stack retail analytics dashboard** offering **real-time sales forecasting**, **interactive visualizations**, and **ML-driven predictions** — all inside a streamlined Streamlit app.

Designed for **data analysts, store managers, consultants, and business owners** to analyze trends, optimize decisions, and forecast future revenue.

---

## 🚀 Key Features

| Feature | Description |
|--------|-------------|
| **Interactive Dashboard** | Live KPIs, filters, drill-downs |
| **Advanced Insights** | Feature importance, outlet performance, product trends |
| **Time Series Forecasting** | Custom date ranges & future sales prediction |
| **Sales Predictor** | Single input form or **batch CSV upload** |
| **Export Reports** | One-click **PDF** or **Excel** export |
| **Powered by Plotly + Streamlit** | Fast, modern, responsive UI |

---

## 🤖 Machine Learning Model  
- **Algorithm:** `XGBoost Regressor`  
- **Key Features Used:**  
  - `Item_MRP`  
  - `Outlet_Identifier`  
  - `Outlet_Size`  
  - `Outlet_Type`  
  - `Outlet_Age` (derived from establishment year)  
- Includes **custom scoring** for business-friendly interpretability.

---

## 🛠️ Installation (Local)

```bash
git clone https://github.com/ratsy10/my-retail-forecast.git
cd my-retail-forecast
pip install -r requirements.txt
streamlit run app.py
```

App will run at:  
👉 **http://localhost:8501**

---

## 🧪 Try a Sample Prediction

### **Single Input (Form)**  
- Item MRP: `150`  
- Outlet Type: `Supermarket Type2`  
- Outlet Size: `Medium`  
- Established: `2005`  

---

### **Batch CSV Upload Example**

```csv
Item_MRP,Outlet_Identifier,Outlet_Size,Outlet_Type,Outlet_Establishment_Year
199.0,OUT017,Medium,Supermarket Type1,2004
249.5,OUT049,High,Supermarket Type3,1999
```

Upload this CSV in the **Batch Prediction** panel.

---

## ☁ Deployment Options

| Platform | Notes |
|---------|-------|
| **Streamlit Community Cloud** | One-click deploy; connect GitHub repo |
| **Render** | Free tier available |
| **Docker / Heroku** | Optional; Dockerfile included |



## 🤝 Contributing  
Pull requests are welcome.  
For major changes, please open an issue first.


