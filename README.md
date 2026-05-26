---
title: Industrial DAE Dashboard
emoji: 🏭
colorFrom: blue
colorTo: slate
sdk: streamlit
app_file: app.py
pinned: false
---

# Industrial DAE — Multi X-Y Dashboard

An advanced Denoising Autoencoder (DAE) platform for sensor reconstruction and KPI prediction in industrial processes.

## Features
- **Multi-X to Multi-Y Forecasting**: Predict complex KPIs from multiple sensor inputs.
- **Denoising Autoencoder**: Robust handling of sensor noise and missing values.
- **Sensitivity Analysis**: Explore "What-If" scenarios to identify plant drivers.
- **Premium UI**: Modern, sleek interface designed for industrial engineers.

## Deployment on Hugging Face Spaces
1. Create a new Space.
2. Select **Streamlit** as the SDK.
3. Upload `app.py`, `requirements.txt`, and your dataset (e.g., `Data_DAE.xlsx`).
4. The app will automatically build and deploy.

## Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```
