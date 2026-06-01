"""
src/ui/pages/upload.py
=======================
Renders the "Upload Data" page.

Allows users to:
  - Upload a new Excel file (auto-saved to SQLite)
  - Load a previously stored dataset from the database
  - View / delete stored datasets
  - Preview the currently active dataset
"""
from __future__ import annotations

import streamlit as st

from src.data.database import (
    delete_dataset_from_db,
    list_datasets_from_db,
    load_dataset_from_db,
    save_dataset_to_db,
)


@st.cache_data
def _read_excel(file_bytes: bytes):
    """Cache the Excel parsing step to avoid re-reading on every rerun."""
    import io
    import pandas as pd
    return pd.read_excel(io.BytesIO(file_bytes))


@st.cache_data
def _read_csv(file_bytes: bytes):
    """Cache the CSV parsing step to avoid re-reading on every rerun."""
    import io
    import pandas as pd
    return pd.read_csv(io.BytesIO(file_bytes))


def render() -> None:
    st.title("Upload Data")

    col1, col2 = st.columns(2)

    # ------------------------------------------------------------------ #
    # Upload new file
    # ------------------------------------------------------------------ #
    with col1:
        st.subheader("Upload New File")
        uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            if uploaded_file.name.lower().endswith(".csv"):
                df = _read_csv(file_bytes)
            else:
                df = _read_excel(file_bytes)
            save_dataset_to_db(uploaded_file.name, df)
            st.session_state.df = df
            st.session_state.data_history[uploaded_file.name] = df
            st.success(
                f"✅ Data saved to database as **{uploaded_file.name}**!"
            )

    # ------------------------------------------------------------------ #
    # Load from database
    # ------------------------------------------------------------------ #
    with col2:
        st.subheader("Load from Database")
        db_datasets = list_datasets_from_db()

        if db_datasets:
            dataset_names = [r[0] for r in db_datasets]
            history_file  = st.selectbox(
                "Select previously uploaded data", dataset_names
            )
            if st.button("Load Selected Data"):
                loaded_df = load_dataset_from_db(history_file)
                if loaded_df is not None:
                    st.session_state.df = loaded_df
                    st.session_state.data_history[history_file] = loaded_df
                    st.success(
                        f"Data switched to **{history_file}** successfully!"
                    )
                else:
                    st.error(
                        f"Could not load **{history_file}** — the stored data is "
                        "incompatible with the current PyArrow version. "
                        "Please delete this entry below and re-upload the file."
                    )
        else:
            st.info(
                "No datasets in database yet. Upload a file to get started."
            )

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Database inventory + delete
    # ------------------------------------------------------------------ #
    db_datasets = list_datasets_from_db()
    if db_datasets:
        st.subheader("📦 Database Inventory")
        import pandas as pd

        inv_df = pd.DataFrame(
            db_datasets,
            columns=["Dataset Name", "Uploaded On", "Rows", "Columns"],
        )
        st.dataframe(inv_df, use_container_width=True)

        del_name = st.selectbox(
            "Select dataset to delete",
            [r[0] for r in db_datasets],
            key="del_ds",
        )
        if st.button("🗑️ Delete Selected Dataset"):
            delete_dataset_from_db(del_name)
            if del_name in st.session_state.data_history:
                del st.session_state.data_history[del_name]
            st.success(f"Deleted **{del_name}** from database.")
            st.rerun()

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Current data preview
    # ------------------------------------------------------------------ #
    if st.session_state.df is not None:
        st.subheader("Current Data Overview")
        st.dataframe(st.session_state.df.head())
        st.write(f"**Shape:** {st.session_state.df.shape}")
