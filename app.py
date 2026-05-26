import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
import io
import os
import sqlite3
import datetime

# ---- SQLite Database Helper Functions ----
DB_PATH = "dashboard.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        upload_time TEXT,
        num_rows INTEGER,
        num_cols INTEGER,
        data BLOB
    )''')
    conn.commit()
    conn.close()

def save_dataset_to_db(name, df):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    blob = df.to_parquet()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT OR REPLACE INTO datasets (name, upload_time, num_rows, num_cols, data)
                 VALUES (?, ?, ?, ?, ?)''', (name, now, len(df), len(df.columns), blob))
    conn.commit()
    conn.close()

def list_datasets_from_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT name, upload_time, num_rows, num_cols FROM datasets ORDER BY upload_time DESC')
    rows = c.fetchall()
    conn.close()
    return rows

def load_dataset_from_db(name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT data FROM datasets WHERE name = ?', (name,))
    row = c.fetchone()
    conn.close()
    if row:
        return pd.read_parquet(io.BytesIO(row[0]))
    return None

def delete_dataset_from_db(name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM datasets WHERE name = ?', (name,))
    conn.commit()
    conn.close()

init_db()

# ---- Model Persistence Helpers ----
import pickle

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def save_model_to_disk(model, scaler_x, scaler_y, x_cols, y_cols, model_name):
    """Save model weights, scalers, and column config to disk."""
    save_path = os.path.join(MODEL_DIR, model_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Save model state dict
    torch.save({
        'state_dict': model.state_dict(),
        'input_dim': len(x_cols),
        'latent_dim': model.encoder[-1].out_features,
        'output_dim': model.predictor[-1].out_features,
    }, os.path.join(save_path, 'model.pth'))
    
    # Save scalers
    with open(os.path.join(save_path, 'scaler_x.pkl'), 'wb') as f:
        pickle.dump(scaler_x, f)
    with open(os.path.join(save_path, 'scaler_y.pkl'), 'wb') as f:
        pickle.dump(scaler_y, f)
    
    # Save column config
    with open(os.path.join(save_path, 'columns.pkl'), 'wb') as f:
        pickle.dump({'x_cols': x_cols, 'y_cols': y_cols}, f)
    
    # Save metadata
    meta = {
        'name': model_name,
        'saved_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_dim': len(x_cols),
        'output_dim': len(y_cols),
        'x_cols': x_cols,
        'y_cols': y_cols,
    }
    with open(os.path.join(save_path, 'metadata.pkl'), 'wb') as f:
        pickle.dump(meta, f)

def list_saved_models():
    """List all saved model directories with their metadata."""
    models = []
    if not os.path.exists(MODEL_DIR):
        return models
    for name in os.listdir(MODEL_DIR):
        meta_path = os.path.join(MODEL_DIR, name, 'metadata.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            models.append(meta)
    return models

def load_model_from_disk(model_name):
    """Load model, scalers, and column config from disk."""
    load_path = os.path.join(MODEL_DIR, model_name)
    
    checkpoint = torch.load(os.path.join(load_path, 'model.pth'), weights_only=False)
    model = IndustrialDAE(
        input_dim=checkpoint['input_dim'],
        latent_dim=checkpoint['latent_dim'],
        output_dim=checkpoint['output_dim']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    with open(os.path.join(load_path, 'scaler_x.pkl'), 'rb') as f:
        scaler_x = pickle.load(f)
    with open(os.path.join(load_path, 'scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)
    with open(os.path.join(load_path, 'columns.pkl'), 'rb') as f:
        cols = pickle.load(f)
    
    return model, scaler_x, scaler_y, cols['x_cols'], cols['y_cols']

st.set_page_config(page_title="Multi X-Y | Industrial DAE", layout="wide", initial_sidebar_state="expanded")

# ---- Premium CSS Styling ----
st.markdown("""
    <style>
        /* Modern Industrial Theme */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@400;600;800&display=swap');
        
        :root {
            --primary: #4da6ff;
            --secondary: #2b6cb0;
            --bg-dark: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --accent: #10b981;
        }

        .main {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f8fafc;
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3, h4 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
        }

        .stButton>button {
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.6rem 1.5rem !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
            width: 100% !important;
        }

        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
            background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
        }

        .stDataFrame, .stTable {
            background-color: var(--card-bg) !important;
            border-radius: 15px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            padding: 10px !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: var(--primary) !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }

        .st-emotion-cache-16idsys p {
            color: #94a3b8 !important;
        }

        /* Custom Cards */
        .status-card {
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)


# Session State Initialization
if 'df' not in st.session_state: st.session_state.df = None
if 'data_history' not in st.session_state: st.session_state.data_history = {}
if 'x_cols' not in st.session_state: st.session_state.x_cols = []
if 'y_cols' not in st.session_state: st.session_state.y_cols = []
if 'X_train' not in st.session_state: st.session_state.X_train = None
if 'X_test' not in st.session_state: st.session_state.X_test = None
if 'y_train' not in st.session_state: st.session_state.y_train = None
if 'y_test' not in st.session_state: st.session_state.y_test = None
if 'scaler_x' not in st.session_state: st.session_state.scaler_x = None
if 'scaler_y' not in st.session_state: st.session_state.scaler_y = None
if 'model_trained' not in st.session_state: st.session_state.model_trained = False
if 'history' not in st.session_state: st.session_state.history = []
if 'sim_history' not in st.session_state: st.session_state.sim_history = []
if 'loaded_sim' not in st.session_state: st.session_state.loaded_sim = None

# Helper Classes for PyTorch
class IndustrialDAE(nn.Module):
    def __init__(self, input_dim=41, latent_dim=15, output_dim=5, dropout_rate=0.2):
        super(IndustrialDAE, self).__init__()

        # --- ENCODER: Learns the "Hidden Physics" ---
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim) # Compressed State
        )

        # --- DECODER: Reconstructs/Heals all features ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim) 
        )

        # --- PREDICTOR: Specifically targets the KPIs ---
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim) 
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        predicted_y = self.predictor(z)
        return reconstructed_x, predicted_y

# Sidebar Navigation
with st.sidebar:
    st.markdown("<h2 style='text-align: left; margin-bottom: 0px;'>Multi X-Y</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: left; color: #4da6ff; margin-top: 0px;'>ML Dashboard</h4>", unsafe_allow_html=True)
    st.markdown("---")
    selected = option_menu(
        menu_title=None,
        options=["Overview", "Upload Data", "Preprocess", "Train Model", "Predict", "What-If", "History", "Comparison"],
        icons=["graph-up", "upload", "gear", "diagram-3", "graph-up-arrow", "magic", "clock-history", "bar-chart"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#2d3748"},
            "nav-link-selected": {"background-color": "#2b6cb0"},
        }
    )

if selected == "Overview":
    st.title("🏭 Industrial DAE — Multi X-Y Dashboard")
    st.caption("End-to-end Denoising Autoencoder for Sensor Reconstruction & KPI Prediction")
    
    # --- System Status Cards ---
    st.markdown("### 🔄 System Status")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        if st.session_state.df is not None:
            st.success(f"✅ Data Loaded\n\n**{st.session_state.df.shape[0]}** rows × **{st.session_state.df.shape[1]}** cols")
        else:
            st.warning("⚠️ No Data\n\nUpload data to begin")
    with s2:
        if len(st.session_state.x_cols) > 0:
            st.success(f"✅ Preprocessed\n\n**{len(st.session_state.x_cols)}** X  |  **{len(st.session_state.y_cols)}** Y")
        else:
            st.warning("⚠️ Not Preprocessed")
    with s3:
        if st.session_state.model_trained:
            st.success("✅ Model Trained\n\nReady for Prediction")
        else:
            st.warning("⚠️ No Model\n\nTrain a model first")
    with s4:
        saved_models_list = list_saved_models()
        st.info(f"💾 Saved Models\n\n**{len(saved_models_list)}** model(s) on disk")
    
    st.markdown("---")
    
    # --- Current Model Performance ---
    if st.session_state.model_trained and st.session_state.X_test is not None:
        st.markdown("### 📊 Current Model Performance")
        
        model = st.session_state.model
        scaler_y = st.session_state.scaler_y
        X_test_t = torch.tensor(st.session_state.X_test, dtype=torch.float32)
        y_test = st.session_state.y_test_raw
        
        model.eval()
        with torch.no_grad():
            _, preds_test_scaled = model(X_test_t)
            preds_test = scaler_y.inverse_transform(preds_test_scaled.numpy())
        
        # Per-feature KPI cards
        kpi_cols = st.columns(len(st.session_state.y_cols))
        r2_vals = []
        for i, col in enumerate(st.session_state.y_cols):
            r2_val = r2_score(y_test[col], preds_test[:, i])
            mae_val = mean_absolute_error(y_test[col], preds_test[:, i])
            r2_vals.append(r2_val)
            with kpi_cols[i]:
                if r2_val >= 0.90: emoji = "🟢"
                elif r2_val >= 0.75: emoji = "🟡"
                else: emoji = "🔴"
                st.metric(label=f"{emoji} {col}", value=f"R² = {r2_val:.4f}", delta=f"MAE = {mae_val:.4f}")
        
        avg_r2 = np.mean(r2_vals)
        if avg_r2 >= 0.90: grade = "Excellent 🟢"
        elif avg_r2 >= 0.75: grade = "Good 🟡"
        else: grade = "Needs Improvement 🔴"
        
        st.markdown(f"**Overall Average R²:** `{avg_r2:.4f}` — **{grade}**")
        
        st.markdown("---")
        
        # Feature lists
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("**Input Features (X)**")
            for c in st.session_state.x_cols:
                st.markdown(f"- `{c}`")
        with col_info2:
            st.markdown("**Target Features (Y)**")
            for c in st.session_state.y_cols:
                st.markdown(f"- `{c}`")
    else:
        st.info("Upload data, preprocess, and train a model to see performance metrics here.")
    
    st.markdown("---")
    
    # --- Database & Saved Models Summary ---
    db_col1, db_col2 = st.columns(2)
    with db_col1:
        st.markdown("### 📦 Datasets in Database")
        db_ds = list_datasets_from_db()
        if len(db_ds) > 0:
            inv_df = pd.DataFrame(db_ds, columns=['Name', 'Uploaded', 'Rows', 'Cols'])
            st.dataframe(inv_df, width='stretch')
        else:
            st.caption("No datasets stored yet.")
    
    with db_col2:
        st.markdown("### 💾 Saved Models")
        if len(saved_models_list) > 0:
            model_df = pd.DataFrame(saved_models_list)[['name', 'saved_at', 'input_dim', 'output_dim']]
            model_df.columns = ['Name', 'Saved At', 'X Features', 'Y Targets']
            st.dataframe(model_df, width='stretch')
        else:
            st.caption("No models saved yet.")
    
    st.markdown("---")
    
    # --- Workflow Guide ---
    st.markdown("### 🗺️ Workflow Guide")
    st.markdown("""
    | Step | Tab | Action |
    |------|-----|--------|
    | 1 | **Upload Data** | Upload Excel dataset or load from database |
    | 2 | **Preprocess** | Select X/Y features, impute missing data, handle outliers |
    | 3 | **Train Model** | Configure hyperparameters, train DAE, or load a saved model |
    | 4 | **Predict** | Evaluate on test data — metrics, scatter plots, residual analysis |
    | 5 | **What-If** | Sensitivity analysis with step changes & trend detection |
    | 6 | **History** | Review all training runs |
    | 7 | **Comparison** | Compare metrics across different model runs |
    """)

elif selected == "Upload Data":
    st.title("Upload Data")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Upload New File")
        uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
        
        if uploaded_file is not None:
            @st.cache_data
            def load_data_from_bytes(file_bytes):
                return pd.read_excel(file_bytes)
            
            df = load_data_from_bytes(uploaded_file)
            save_dataset_to_db(uploaded_file.name, df)
            st.session_state.df = df
            st.session_state.data_history[uploaded_file.name] = df
            st.success(f"✅ Data saved to database as **{uploaded_file.name}**!")

    with col2:
        st.subheader("Load from Database")
        db_datasets = list_datasets_from_db()
        
        if len(db_datasets) > 0:
            dataset_names = [r[0] for r in db_datasets]
            history_file = st.selectbox("Select previously uploaded data", dataset_names)
            if st.button("Load Selected Data"):
                loaded_df = load_dataset_from_db(history_file)
                if loaded_df is not None:
                    st.session_state.df = loaded_df
                    st.session_state.data_history[history_file] = loaded_df
                    st.success(f"Data switched to **{history_file}** successfully!")
                else:
                    st.error("Failed to load dataset from database.")
        else:
            st.info("No datasets in database yet. Upload a file to get started.")

    st.markdown("---")
    
    # Show database inventory
    db_datasets = list_datasets_from_db()
    if len(db_datasets) > 0:
        st.subheader("📦 Database Inventory")
        inv_df = pd.DataFrame(db_datasets, columns=['Dataset Name', 'Uploaded On', 'Rows', 'Columns'])
        st.dataframe(inv_df, width='stretch')
        
        del_name = st.selectbox("Select dataset to delete", [r[0] for r in db_datasets], key="del_ds")
        if st.button("🗑️ Delete Selected Dataset"):
            delete_dataset_from_db(del_name)
            if del_name in st.session_state.data_history:
                del st.session_state.data_history[del_name]
            st.success(f"Deleted **{del_name}** from database.")
            st.rerun()

    st.markdown("---")
    if st.session_state.df is not None:
        st.subheader("Current Data Overview")
        st.dataframe(st.session_state.df.head())
        st.write(f"**Shape:** {st.session_state.df.shape}")

elif selected == "Preprocess":
    st.title("Preprocess Data")
    db_datasets = list_datasets_from_db()
    
    if len(db_datasets) > 0:
        col1, col2 = st.columns([3, 1])
        with col1:
            history_file_prep = st.selectbox("Select Active Dataset", [r[0] for r in db_datasets], key="prep_dataset")
        with col2:
            st.write("")
            st.write("")
            if st.button("Load Dataset", key="load_prep"):
                loaded_df = load_dataset_from_db(history_file_prep)
                if loaded_df is not None:
                    st.session_state.df = loaded_df
                    st.session_state.data_history[history_file_prep] = loaded_df
                    st.success(f"Dataset switched to {history_file_prep}")
                    st.rerun()
        st.markdown("---")
        
    if st.session_state.df is None:
        st.warning("Please upload data first in the 'Upload Data' tab.")
    else:
        df = st.session_state.df
        
        # Force conversion of object columns to numeric (coercing messy strings to NaN)
        # This fixes issues where sensor data is accidentally parsed as text
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        st.subheader("Variable Selection")
        
        col_x, col_y = st.columns(2)
        
        with col_x:
            st.markdown("**Select Input Features (X)**")
            # Select All / Deselect All for X
            select_all_x = st.checkbox("Select All X", value=len(st.session_state.x_cols) == len(numeric_cols), key="sel_all_x")
            x_cols = []
            for col in numeric_cols:
                default_checked = (col in st.session_state.x_cols) if not select_all_x else True
                if st.checkbox(col, value=default_checked, key=f"x_{col}"):
                    x_cols.append(col)
        
        with col_y:
            st.markdown("**Select Target Variables (Y)**")
            y_options = [c for c in numeric_cols if c not in x_cols]
            select_all_y = st.checkbox("Select All Y", value=len(st.session_state.y_cols) == len(y_options) and len(y_options) > 0, key="sel_all_y")
            y_cols = []
            for col in y_options:
                default_checked = (col in st.session_state.y_cols) if not select_all_y else True
                if st.checkbox(col, value=default_checked, key=f"y_{col}"):
                    y_cols.append(col)
        
        st.subheader("Missing Data Imputation & Outliers")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            imputation_method = st.selectbox("Missing Value Imputation Method", ["Mean", "Median", "Zero"])
        with col_f2:
            outlier_method = st.radio("Select Outlier Treatment Method", ["None", "IQR Capping", "Min-Max Percentile Capping (1% - 99%)"])
        
        # --- Custom Min-Max Filter ---
        st.subheader("🔧 Custom Min-Max Filter (Per Tag)")
        st.caption("Select specific features and set custom min/max bounds. Data outside these limits will be clipped.")
        
        all_selected = x_cols + y_cols
        custom_filter_tags = st.multiselect("Select Tags to Apply Custom Min-Max Filter", all_selected, default=[], key="custom_filter_tags")
        
        custom_filters = {}
        if len(custom_filter_tags) > 0:
            filter_cols = st.columns(3)
            for idx, tag in enumerate(custom_filter_tags):
                tag_min = float(df[tag].min())
                tag_max = float(df[tag].max())
                with filter_cols[idx % 3]:
                    st.markdown(f"**{tag}**")
                    st.caption(f"Data Range: {tag_min:.4f} — {tag_max:.4f}")
                    c1, c2 = st.columns(2)
                    with c1:
                        user_min = st.number_input(f"Min", value=tag_min, format="%.4f", key=f"fmin_{tag}")
                    with c2:
                        user_max = st.number_input(f"Max", value=tag_max, format="%.4f", key=f"fmax_{tag}")
                    custom_filters[tag] = {"min": user_min, "max": user_max}
        
        if st.button("Apply Preprocessing"):
            if len(x_cols) == 0 or len(y_cols) == 0:
                st.error("Please select at least one X and one Y variable.")
            else:
                st.session_state.x_cols = x_cols
                st.session_state.y_cols = y_cols
                
                data_x = df[x_cols].copy()
                data_y = df[y_cols].copy()
                
                # Show Feature Statistics
                st.markdown("### Feature-wise Statistics (Before Imputation)")
                stats_df = pd.DataFrame({
                    'Missing Count': data_x.isnull().sum(),
                    'Missing %': (data_x.isnull().sum() / len(data_x) * 100).round(2),
                    'Min': data_x.min(),
                    'Mean': data_x.mean(),
                    'Max': data_x.max()
                })
                st.dataframe(stats_df)
                
                # Impute NaNs
                if imputation_method == "Mean":
                    data_x = data_x.fillna(data_x.mean())
                    data_y = data_y.fillna(data_y.mean())
                elif imputation_method == "Median":
                    data_x = data_x.fillna(data_x.median())
                    data_y = data_y.fillna(data_y.median())
                elif imputation_method == "Zero":
                    data_x = data_x.fillna(0)
                    data_y = data_y.fillna(0)
                
                # 4. Outlier Handling
                if outlier_method == "IQR Capping":
                    for col in data_x.columns:
                        Q1 = data_x[col].quantile(0.25)
                        Q3 = data_x[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        data_x[col] = np.clip(data_x[col], lower_bound, upper_bound)
                    for col in data_y.columns:
                        Q1 = data_y[col].quantile(0.25)
                        Q3 = data_y[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        data_y[col] = np.clip(data_y[col], lower_bound, upper_bound)
                elif outlier_method == "Min-Max Percentile Capping (1% - 99%)":
                    for col in data_x.columns:
                        lower_bound = data_x[col].quantile(0.01)
                        upper_bound = data_x[col].quantile(0.99)
                        data_x[col] = np.clip(data_x[col], lower_bound, upper_bound)
                    for col in data_y.columns:
                        lower_bound = data_y[col].quantile(0.01)
                        upper_bound = data_y[col].quantile(0.99)
                        data_y[col] = np.clip(data_y[col], lower_bound, upper_bound)
                
                # 5. Apply Custom Min-Max Filters
                for tag, bounds in custom_filters.items():
                    if tag in data_x.columns:
                        data_x[tag] = np.clip(data_x[tag], bounds['min'], bounds['max'])
                    if tag in data_y.columns:
                        data_y[tag] = np.clip(data_y[tag], bounds['min'], bounds['max'])
                
                st.markdown("### Feature-wise Statistics (After Preprocessing)")
                stats_after_df = pd.DataFrame({
                    'Missing Count': data_x.isnull().sum(),
                    'Missing %': (data_x.isnull().sum() / len(data_x) * 100).round(2),
                    'Min': data_x.min(),
                    'Mean': data_x.mean(),
                    'Max': data_x.max()
                })
                st.dataframe(stats_after_df)
                
                X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
                
                scaler_x = StandardScaler()
                scaler_y = StandardScaler()
                
                st.session_state.X_train = scaler_x.fit_transform(X_train)
                st.session_state.X_test = scaler_x.transform(X_test)
                st.session_state.y_train = scaler_y.fit_transform(y_train)
                st.session_state.y_test = scaler_y.transform(y_test)
                
                st.session_state.scaler_x = scaler_x
                st.session_state.scaler_y = scaler_y
                st.session_state.y_test_raw = y_test
                
                st.success(f"Preprocessing complete! Applied {outlier_method}. Train/Test split created and features scaled.")

elif selected == "Train Model":
    st.title("Train Model (Industrial DAE)")
    
    # --- Load a previously saved model ---
    saved_models = list_saved_models()
    if len(saved_models) > 0:
        with st.expander("📂 Load a Previously Saved Model", expanded=False):
            model_meta_df = pd.DataFrame(saved_models)[['name', 'saved_at', 'input_dim', 'output_dim']]
            model_meta_df.columns = ['Model Name', 'Saved At', 'Input Features', 'Output Targets']
            st.dataframe(model_meta_df, width='stretch')
            
            sel_model_name = st.selectbox("Select Model to Load", [m['name'] for m in saved_models])
            if st.button("Load Selected Model"):
                loaded_model, loaded_sx, loaded_sy, loaded_x, loaded_y = load_model_from_disk(sel_model_name)
                st.session_state.model = loaded_model
                st.session_state.scaler_x = loaded_sx
                st.session_state.scaler_y = loaded_sy
                st.session_state.x_cols = loaded_x
                st.session_state.y_cols = loaded_y
                st.session_state.model_trained = True
                st.success(f"✅ Model **{sel_model_name}** loaded! You can now use Predict and What-If tabs.")
                st.rerun()
        st.markdown("---")
    
    if st.session_state.X_train is None:
        st.warning("Please preprocess data first in the 'Preprocess' tab.")
    else:
        st.subheader("Hyperparameters")
        col1, col2 = st.columns(2)
        with col1:
            masking_ratio = st.slider("Masking Ratio (Corruption)", 0.0, 0.5, 0.10)
            epochs = st.number_input("Epochs", 10, 1000, 150)
            lr = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
            auto_train = st.checkbox("Auto-Train (Until R2 > 0.85 & MAE lower)", value=False)
        with col2:
            latent_dim = st.slider("Latent Dimension", 2, max(2, len(st.session_state.x_cols)), 15)
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
            weight_to_pred = st.number_input("Weight to Predictor Loss", 0.1, 10.0, 5.0)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128, 256], index=3)
            
        if st.button("Train"):
            X_train_t = torch.tensor(st.session_state.X_train, dtype=torch.float32)
            y_train_t = torch.tensor(st.session_state.y_train, dtype=torch.float32)
            X_test_t = torch.tensor(st.session_state.X_test, dtype=torch.float32)
            y_test_t = torch.tensor(st.session_state.y_test, dtype=torch.float32)
            
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            input_dim = X_train_t.shape[1]
            output_dim = y_train_t.shape[1]
            
            model = IndustrialDAE(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim, dropout_rate=dropout_rate)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion_recon = nn.MSELoss()
            criterion_pred = nn.HuberLoss()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            epoch_recon_losses = []
            epoch_pred_losses = []
            val_recon_losses = []
            val_pred_losses = []
            
            y_test_raw = st.session_state.y_test_raw
            scaler_y = st.session_state.scaler_y
            
            max_train_epochs = 2000 if auto_train else epochs
            best_r2 = -float('inf')
            best_mae = float('inf')
            
            for epoch in range(max_train_epochs):
                model.train()
                batch_recon_loss = 0
                batch_pred_loss = 0
                
                for batch_x, batch_y in train_loader:
                    clean_x = batch_x
                    
                    # Apply Masking
                    random_probabilities = torch.rand(clean_x.shape)
                    mask = random_probabilities < masking_ratio
                    noised_x = clean_x.clone()
                    noised_x[mask] = 0.0
                    
                    recon_x, pred_y = model(noised_x)
                    
                    loss_recon = criterion_recon(recon_x, clean_x)
                    loss_pred = criterion_pred(pred_y, batch_y)
                    total_loss = loss_recon + (weight_to_pred * loss_pred)
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    batch_recon_loss += loss_recon.item()
                    batch_pred_loss += loss_pred.item()
                
                epoch_recon_losses.append(batch_recon_loss / len(train_loader))
                epoch_pred_losses.append(batch_pred_loss / len(train_loader))
                
                # Validation Pass
                model.eval()
                with torch.no_grad():
                    # Evaluate on clean test set without masking (standard practice for evaluation)
                    val_recon, val_pred = model(X_test_t)
                    v_loss_recon = criterion_recon(val_recon, X_test_t)
                    v_loss_pred = criterion_pred(val_pred, y_test_t)
                    val_recon_losses.append(v_loss_recon.item())
                    val_pred_losses.append(v_loss_pred.item())
                
                if auto_train:
                    if (epoch + 1) % 10 == 0:
                        with torch.no_grad():
                            preds_test_scaled = val_pred
                            preds_test = scaler_y.inverse_transform(preds_test_scaled.numpy())
                        
                        r2_vals = [r2_score(y_test_raw[col], preds_test[:, i]) for i, col in enumerate(st.session_state.y_cols)]
                        mae_vals = [mean_absolute_error(y_test_raw[col], preds_test[:, i]) for i, col in enumerate(st.session_state.y_cols)]
                        avg_r2 = np.mean(r2_vals)
                        avg_mae = np.mean(mae_vals)
                        
                        status_text.text(f"Auto-Training... Epoch {epoch+1} | Avg R2: {avg_r2:.4f} | Avg MAE: {avg_mae:.4f}")
                        
                        if avg_r2 > 0.85 and avg_mae <= best_mae:
                            status_text.text(f"Reached Target! Stopped at Epoch {epoch+1} with Avg R2 = {avg_r2:.4f}, Avg MAE = {avg_mae:.4f}")
                            break
                        
                        if avg_r2 > best_r2: best_r2 = avg_r2
                        if avg_mae < best_mae: best_mae = avg_mae
                else:
                    progress_bar.progress((epoch + 1) / epochs)
            
            if not auto_train:
                status_text.text("Training Complete!")
                
            st.session_state.model = model
            st.session_state.model_trained = True
            
            # Final Evaluation
            model.eval()
            with torch.no_grad():
                _, val_pred = model(X_test_t)
                preds_test = scaler_y.inverse_transform(val_pred.numpy())
            
            metrics_df = pd.DataFrame(index=st.session_state.y_cols, columns=['RMSE', 'MAE', 'R2 Score'])
            for i, col in enumerate(st.session_state.y_cols):
                mse = mean_squared_error(y_test_raw[col], preds_test[:, i])
                metrics_df.loc[col, 'RMSE'] = np.sqrt(mse)
                metrics_df.loc[col, 'MAE'] = mean_absolute_error(y_test_raw[col], preds_test[:, i])
                metrics_df.loc[col, 'R2 Score'] = r2_score(y_test_raw[col], preds_test[:, i])
            
            avg_rmse = metrics_df['RMSE'].mean()
            
            run_id = len(st.session_state.history) + 1
            st.session_state.history.append({
                "Run ID": run_id,
                "Masking": masking_ratio,
                "Latent Dim": latent_dim,
                "Epochs": len(epoch_pred_losses),
                "Avg Test RMSE": avg_rmse,
                "Model": model
            })
            
            # Auto-save model to disk
            model_name = f"DAE_Run{run_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            save_model_to_disk(model, st.session_state.scaler_x, st.session_state.scaler_y,
                               st.session_state.x_cols, st.session_state.y_cols, model_name)
            
            st.success(f"✅ Model trained, saved as **{model_name}**, and added to History! (Epochs: {len(epoch_pred_losses)})")
            st.subheader("Training Post-Evaluation Metrics")
            st.dataframe(metrics_df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("DAE Reconstruction Loss (MSE)")
                fig, ax = plt.subplots()
                ax.plot(epoch_recon_losses, color='blue', label='Train Loss')
                ax.plot(val_recon_losses, color='cyan', label='Validation Loss')
                ax.legend()
                st.pyplot(fig)
            with col2:
                st.subheader("Predictor Loss (Huber)")
                fig, ax = plt.subplots()
                ax.plot(epoch_pred_losses, color='orange', label='Train Loss')
                ax.plot(val_pred_losses, color='red', label='Validation Loss')
                ax.legend()
                st.pyplot(fig)

elif selected == "Predict":
    st.title("Predict & Evaluate")
    if not st.session_state.model_trained:
        st.warning("Please train the model first in the 'Train Model' tab.")
    else:
        model = st.session_state.model
        scaler_y = st.session_state.scaler_y
        X_test_t = torch.tensor(st.session_state.X_test, dtype=torch.float32)
        y_test = st.session_state.y_test_raw
        
        model.eval()
        with torch.no_grad():
            _, val_pred = model(X_test_t)
            preds_test = scaler_y.inverse_transform(val_pred.numpy())
            
        st.subheader("Test Set Metrics")
        metrics_df = pd.DataFrame(index=st.session_state.y_cols, columns=['RMSE', 'MAE', 'R2 Score', 'MAPE (%)'])
        for i, col in enumerate(st.session_state.y_cols):
            actual = y_test[col].values
            predicted = preds_test[:, i]
            mse = mean_squared_error(actual, predicted)
            metrics_df.loc[col, 'RMSE'] = np.sqrt(mse)
            metrics_df.loc[col, 'MAE'] = mean_absolute_error(actual, predicted)
            metrics_df.loc[col, 'R2 Score'] = r2_score(actual, predicted)
            # MAPE - handle zeros
            nonzero_mask = actual != 0
            if nonzero_mask.sum() > 0:
                metrics_df.loc[col, 'MAPE (%)'] = np.mean(np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])) * 100
            else:
                metrics_df.loc[col, 'MAPE (%)'] = 0.0
        st.dataframe(metrics_df, width='stretch')
        
        # --- KPI Summary Cards ---
        st.subheader("📊 Model Performance Summary")
        kpi_cols = st.columns(len(st.session_state.y_cols))
        for i, col in enumerate(st.session_state.y_cols):
            r2_val = float(metrics_df.loc[col, 'R2 Score'])
            mae_val = float(metrics_df.loc[col, 'MAE'])
            with kpi_cols[i]:
                if r2_val >= 0.90:
                    emoji = "🟢"
                elif r2_val >= 0.75:
                    emoji = "🟡"
                else:
                    emoji = "🔴"
                st.metric(label=f"{emoji} {col}", value=f"R² = {r2_val:.4f}", delta=f"MAE = {mae_val:.4f}")
        
        # --- Actual vs Predicted Line Charts ---
        st.subheader("📈 Actual vs Predicted (All Y Features)")
        pts = min(100, len(y_test))
        
        for i, col in enumerate(st.session_state.y_cols):
            r2_val = float(metrics_df.loc[col, 'R2 Score'])
            chart_df = pd.DataFrame({
                'Sample Index': range(pts),
                'Actual': y_test[col].values[:pts],
                'Predicted': preds_test[:pts, i]
            })
            chart_df_melted = chart_df.melt(id_vars=['Sample Index'], value_vars=['Actual', 'Predicted'], var_name='Type', value_name='Value')
            
            fig = px.line(chart_df_melted, x='Sample Index', y='Value', color='Type', 
                          title=f"{col}  |  R² = {r2_val:.4f}")
            fig.update_layout(yaxis=dict(autorange=True))
            st.plotly_chart(fig, width='stretch')
        
        # --- Scatter Plot: Actual vs Predicted with 45° line ---
        st.subheader("🎯 Scatter Plot: Actual vs Predicted")
        scatter_cols = st.columns(min(len(st.session_state.y_cols), 3))
        for i, col in enumerate(st.session_state.y_cols):
            actual = y_test[col].values
            predicted = preds_test[:, i]
            r2_val = float(metrics_df.loc[col, 'R2 Score'])
            with scatter_cols[i % 3]:
                fig = px.scatter(x=actual, y=predicted, labels={'x': 'Actual', 'y': 'Predicted'},
                                 title=f"{col} | R² = {r2_val:.4f}", opacity=0.5)
                # Add 45-degree ideal line
                min_val = min(actual.min(), predicted.min())
                max_val = max(actual.max(), predicted.max())
                fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                              line=dict(color="red", dash="dash", width=2))
                fig.update_layout(yaxis=dict(autorange=True), height=400)
                st.plotly_chart(fig, width='stretch')
        
        # --- Residual Analysis ---
        st.subheader("📉 Residual Analysis (Error Distribution)")
        residual_cols = st.columns(min(len(st.session_state.y_cols), 3))
        for i, col in enumerate(st.session_state.y_cols):
            actual = y_test[col].values
            predicted = preds_test[:, i]
            residuals = actual - predicted
            with residual_cols[i % 3]:
                fig = px.histogram(residuals, nbins=30, title=f"Residuals: {col}",
                                   labels={'value': 'Error (Actual - Predicted)', 'count': 'Frequency'})
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, width='stretch')
        
        # --- Download Predictions ---
        st.subheader("📥 Export Predictions")
        export_df = y_test.copy().reset_index(drop=True)
        for i, col in enumerate(st.session_state.y_cols):
            export_df[f"Predicted_{col}"] = preds_test[:, i]
            export_df[f"Error_{col}"] = y_test[col].values - preds_test[:, i]
        
        csv_pred = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Predictions with Errors (CSV)",
            data=csv_pred,
            file_name="Predictions_with_Errors.csv",
            mime="text/csv",
        )

elif selected == "What-If":
    st.title("What-If Simulator & Sensitivity Analysis")
    if not st.session_state.model_trained:
        st.warning("Please train the model first in the 'Train Model' tab.")
    else:
        df = st.session_state.df
        model = st.session_state.model
        scaler_x = st.session_state.scaler_x
        scaler_y = st.session_state.scaler_y
        
        # --- STEP 1: Select Y targets to observe ---
        st.markdown("### 1. Select Y Targets to Observe")
        target_y_cols = st.multiselect("Select one or more Y features to see impact on",
                                        st.session_state.y_cols, default=st.session_state.y_cols[:1])
        
        if len(target_y_cols) == 0:
            st.warning("Please select at least one Y target.")
        else:
            # --- STEP 2: Configure each X feature as Constant or Vary ---
            st.markdown("### 2. Configure X Features (Constant / Vary)")
            st.caption("For each X feature, choose whether to keep it constant at a fixed value or vary it with a step change.")
            
            feature_config = {}
            num_cols_per_row = 2
            x_cols_list = st.session_state.x_cols
            
            for row_start in range(0, len(x_cols_list), num_cols_per_row):
                row_cols = st.columns(num_cols_per_row)
                for j in range(num_cols_per_row):
                    idx = row_start + j
                    if idx >= len(x_cols_list):
                        break
                    feat = x_cols_list[idx]
                    with row_cols[j]:
                        with st.expander(f"**{feat}**", expanded=False):
                            mode = st.radio(f"Mode for {feat}", ["Constant", "Vary"],
                                            key=f"mode_{feat}", horizontal=True)
                            if mode == "Constant":
                                if st.session_state.loaded_sim is not None and feat in st.session_state.loaded_sim.get('constants', {}):
                                    def_val = float(st.session_state.loaded_sim['constants'][feat])
                                else:
                                    def_val = float(df[feat].mean())
                                val = st.number_input(f"Value for {feat}", value=def_val,
                                                      format="%.4f", key=f"const_{feat}")
                                feature_config[feat] = {"mode": "Constant", "value": val}
                            else:
                                feat_min = float(df[feat].min())
                                feat_max = float(df[feat].max())
                                default_ss = float((feat_max - feat_min) / 20.0)
                                if default_ss == 0: default_ss = 1.0
                                ss = st.number_input(f"Step Size for {feat}", value=default_ss,
                                                     min_value=0.000001, format="%.6f", key=f"step_{feat}")
                                feature_config[feat] = {"mode": "Vary", "step_size": ss,
                                                        "min": feat_min, "max": feat_max}
            
            # --- STEP 3: Run Simulation ---
            if st.button("🚀 Run What-If Simulation"):
                varying_features = {k: v for k, v in feature_config.items() if v["mode"] == "Vary"}
                constant_features = {k: v for k, v in feature_config.items() if v["mode"] == "Constant"}
                
                if len(varying_features) == 0:
                    st.error("Please set at least one X feature to 'Vary' mode.")
                else:
                    # Build sweep arrays for each varying feature
                    sweep_arrays = {}
                    for feat, cfg in varying_features.items():
                        mn, mx, ss = cfg["min"], cfg["max"], cfg["step_size"]
                        if mn == mx:
                            mn -= 1.0
                            mx += 1.0
                        arr = np.arange(mn, mx + ss, ss)
                        if len(arr) > 500:
                            arr = arr[:500]
                        sweep_arrays[feat] = arr
                    
                    # If single varying feature: simple 1D sweep
                    if len(varying_features) == 1:
                        vary_feat = list(varying_features.keys())[0]
                        sweep_vals = sweep_arrays[vary_feat]
                        
                        sim_df = pd.DataFrame()
                        sim_df[vary_feat] = sweep_vals
                        for col in st.session_state.x_cols:
                            if col != vary_feat:
                                sim_df[col] = constant_features[col]["value"]
                        sim_df = sim_df[st.session_state.x_cols]
                        
                        input_scaled = scaler_x.transform(sim_df)
                        input_t = torch.tensor(input_scaled, dtype=torch.float32)
                        
                        model.eval()
                        with torch.no_grad():
                            _, pred_sim_scaled = model(input_t)
                            pred_sim = scaler_y.inverse_transform(pred_sim_scaled.numpy())
                        
                        # Build results
                        results_df = pd.DataFrame({vary_feat: sweep_vals})
                        for ty in target_y_cols:
                            y_idx = st.session_state.y_cols.index(ty)
                            preds = pred_sim[:, y_idx]
                            results_df[f"Predicted {ty}"] = preds
                            # Trend
                            trends = ["-"]
                            for i in range(1, len(preds)):
                                diff = preds[i] - preds[i-1]
                                if diff > 1e-5: trends.append("Increasing 📈")
                                elif diff < -1e-5: trends.append("Decreasing 📉")
                                else: trends.append("Constant ➖")
                            results_df[f"Trend {ty}"] = trends
                        
                        st.markdown(f"### Simulation Results")
                        
                        # Plotly chart for each Y
                        for ty in target_y_cols:
                            fig = px.line(results_df, x=vary_feat, y=f"Predicted {ty}",
                                          title=f"{vary_feat} → {ty}")
                            fig.update_layout(yaxis=dict(autorange=True))
                            st.plotly_chart(fig, width='stretch')
                        
                        st.dataframe(results_df, width='stretch')
                    
                    else:
                        # Multiple varying features: sweep each one independently while others stay at constant/mean
                        st.markdown("### Simulation Results (Per-Feature Sweep)")
                        all_results = []
                        
                        for vary_feat, arr in sweep_arrays.items():
                            sim_df = pd.DataFrame()
                            sim_df[vary_feat] = arr
                            for col in st.session_state.x_cols:
                                if col != vary_feat:
                                    if col in constant_features:
                                        sim_df[col] = constant_features[col]["value"]
                                    elif col in varying_features:
                                        # Other varying features held at their mean during this sweep
                                        sim_df[col] = float(df[col].mean())
                            sim_df = sim_df[st.session_state.x_cols]
                            
                            input_scaled = scaler_x.transform(sim_df)
                            input_t = torch.tensor(input_scaled, dtype=torch.float32)
                            
                            model.eval()
                            with torch.no_grad():
                                _, pred_sim_scaled = model(input_t)
                                pred_sim = scaler_y.inverse_transform(pred_sim_scaled.numpy())
                            
                            for ty in target_y_cols:
                                y_idx = st.session_state.y_cols.index(ty)
                                preds = pred_sim[:, y_idx]
                                
                                trends = ["-"]
                                for i in range(1, len(preds)):
                                    diff = preds[i] - preds[i-1]
                                    if diff > 1e-5: trends.append("Increasing 📈")
                                    elif diff < -1e-5: trends.append("Decreasing 📉")
                                    else: trends.append("Constant ➖")
                                
                                res_df = pd.DataFrame({
                                    vary_feat: arr,
                                    f"Predicted {ty}": preds,
                                    "Trend": trends
                                })
                                all_results.append({"x": vary_feat, "y": ty, "df": res_df})
                                
                                fig = px.line(res_df, x=vary_feat, y=f"Predicted {ty}",
                                              title=f"{vary_feat} → {ty}")
                                fig.update_layout(yaxis=dict(autorange=True))
                                st.plotly_chart(fig, width='stretch')
                        
                        # Combined download
                        combined = pd.DataFrame()
                        for r in all_results:
                            temp = r["df"].copy()
                            temp["Varied X"] = r["x"]
                            temp["Target Y"] = r["y"]
                            combined = pd.concat([combined, temp], ignore_index=True)
                        st.dataframe(combined, use_container_width=True)
                    
                    # Download button
                    if len(varying_features) == 1:
                        csv_data = results_df.to_csv(index=False).encode('utf-8')
                    else:
                        csv_data = combined.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📥 Download Simulation Results (CSV)",
                        data=csv_data,
                        file_name="WhatIf_Simulation_Results.csv",
                        mime="text/csv",
                    )
                    
                    # Save to sim history
                    const_dict = {k: v["value"] for k, v in constant_features.items()}
                    vary_dict = {k: v["step_size"] for k, v in varying_features.items()}
                    st.session_state.sim_history.append({
                        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Varying Features": ", ".join(varying_features.keys()),
                        "Target KPIs": ", ".join(target_y_cols),
                        "Step Sizes": str(vary_dict),
                        "constants": const_dict
                    })
                    st.success("✅ Simulation Run Saved to Action History!")

        st.markdown("---")
        st.markdown("### 🕒 Simulation Action History")
        if len(st.session_state.sim_history) == 0:
            st.info("No actions performed yet. Run a simulation to save it to history.")
        else:
            history_df = pd.DataFrame(st.session_state.sim_history).drop(columns=['constants'])
            st.dataframe(history_df, use_container_width=True)
            
            st.markdown("**Load a Past Action Scenario:**")
            selected_timestamp = st.selectbox("Select Action by Timestamp", [h['Timestamp'] for h in reversed(st.session_state.sim_history)])
            if st.button("Load Selected Scenario"):
                scenario = next(h for h in st.session_state.sim_history if h['Timestamp'] == selected_timestamp)
                st.session_state.loaded_sim = scenario
                st.success(f"Scenario from {selected_timestamp} loaded! The constant feature inputs have been updated.")
                st.rerun()

elif selected == "History":
    st.title("Training History")
    if len(st.session_state.history) == 0:
        st.info("No training history available. Train a model first.")
    else:
        history_df = pd.DataFrame(st.session_state.history).drop(columns=['Model'])
        st.dataframe(history_df)
        
        load_run = st.selectbox("Select a Run ID to load as active model", history_df['Run ID'].tolist())
        if st.button("Load Model"):
            run_data = next(item for item in st.session_state.history if item["Run ID"] == load_run)
            st.session_state.model = run_data["Model"]
            st.session_state.model_trained = True
            st.success(f"Model from Run {load_run} loaded successfully!")

elif selected == "Comparison":
    st.title("Model Comparison")
    if len(st.session_state.history) < 2:
        st.info("Need at least 2 training runs to compare. Go to 'Train Model' and try different hyperparameters.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        
        st.subheader("Average Test Metric Comparison")
        fig, ax = plt.subplots(figsize=(8, 4))
        metric_col = 'Avg Test RMSE' if 'Avg Test RMSE' in history_df.columns else 'Avg Test MSE'
        ax.bar(history_df['Run ID'].astype(str), history_df[metric_col], color='skyblue')
        ax.set_xlabel('Run ID')
        ax.set_ylabel(metric_col)
        st.pyplot(fig)
