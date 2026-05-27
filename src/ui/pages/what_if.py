"""
src/ui/pages/what_if.py
========================
Renders the "What-If Simulator & Sensitivity Analysis" page.

Workflow:
  1. User selects Y targets to observe
  2. User configures each X feature as Constant (fixed value) or Vary
     (sweep from min to max with a user-defined step size)
  3. Clicking "Run What-If Simulation" calls the sweep engine
  4. Results are displayed as Plotly line charts + a data table
  5. Simulation runs are saved to session history and can be reloaded
"""
from __future__ import annotations

import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from src.simulation.what_if import (
    build_sweep_array,
    run_multi_sweep,
    run_single_sweep,
)


def render() -> None:
    st.title("What-If Simulator & Sensitivity Analysis")

    # ------------------------------------------------------------------ #
    # Guard: model must be trained
    # ------------------------------------------------------------------ #
    if not st.session_state.model_trained:
        st.warning("Please train the model first in the 'Train Model' tab.")
        return

    df       = st.session_state.df
    model    = st.session_state.model
    scaler_x = st.session_state.scaler_x
    scaler_y = st.session_state.scaler_y
    x_cols   = st.session_state.x_cols
    y_cols   = st.session_state.y_cols

    # ------------------------------------------------------------------ #
    # Step 1: Select Y targets to observe
    # ------------------------------------------------------------------ #
    st.markdown("### 1. Select Y Targets to Observe")
    target_y_cols = st.multiselect(
        "Select one or more Y features to see impact on",
        y_cols,
        default=y_cols[:1],
    )

    if not target_y_cols:
        st.warning("Please select at least one Y target.")
        return

    # ------------------------------------------------------------------ #
    # Step 2: Configure each X feature (Constant / Vary)
    # ------------------------------------------------------------------ #
    st.markdown("### 2. Configure X Features (Constant / Vary)")
    st.caption(
        "For each X feature, choose whether to keep it constant at a fixed "
        "value or vary it with a step change."
    )

    feature_config: dict = {}
    num_cols_per_row = 2

    for row_start in range(0, len(x_cols), num_cols_per_row):
        row_cols = st.columns(num_cols_per_row)
        for j in range(num_cols_per_row):
            idx = row_start + j
            if idx >= len(x_cols):
                break
            feat = x_cols[idx]
            with row_cols[j]:
                with st.expander(f"**{feat}**", expanded=False):
                    mode = st.radio(
                        f"Mode for {feat}",
                        ["Constant", "Vary"],
                        key=f"mode_{feat}",
                        horizontal=True,
                    )
                    if mode == "Constant":
                        # Pre-fill from loaded scenario if available
                        loaded_sim = st.session_state.loaded_sim
                        def_val = (
                            float(loaded_sim["constants"][feat])
                            if loaded_sim and feat in loaded_sim.get("constants", {})
                            else float(df[feat].mean())
                        )
                        val = st.number_input(
                            f"Value for {feat}",
                            value=def_val,
                            format="%.4f",
                            key=f"const_{feat}",
                        )
                        feature_config[feat] = {"mode": "Constant", "value": val}
                    else:
                        feat_min = float(df[feat].min())
                        feat_max = float(df[feat].max())
                        default_ss = float((feat_max - feat_min) / 20.0)
                        if default_ss == 0:
                            default_ss = 1.0
                        ss = st.number_input(
                            f"Step Size for {feat}",
                            value=default_ss,
                            min_value=0.000001,
                            format="%.6f",
                            key=f"step_{feat}",
                        )
                        feature_config[feat] = {
                            "mode":      "Vary",
                            "step_size": ss,
                            "min":       feat_min,
                            "max":       feat_max,
                        }

    # ------------------------------------------------------------------ #
    # Step 3: Run simulation
    # ------------------------------------------------------------------ #
    if st.button("🚀 Run What-If Simulation"):
        varying_features  = {k: v for k, v in feature_config.items() if v["mode"] == "Vary"}
        constant_features = {k: v for k, v in feature_config.items() if v["mode"] == "Constant"}

        if not varying_features:
            st.error("Please set at least one X feature to 'Vary' mode.")
            return

        # Build sweep arrays
        sweep_arrays = {
            feat: build_sweep_array(cfg["min"], cfg["max"], cfg["step_size"])
            for feat, cfg in varying_features.items()
        }

        # ---- Single-feature sweep ----
        if len(varying_features) == 1:
            vary_feat  = list(varying_features.keys())[0]
            sweep_vals = sweep_arrays[vary_feat]

            results_df = run_single_sweep(
                vary_feat        = vary_feat,
                sweep_vals       = sweep_vals,
                constant_features= constant_features,
                x_cols           = x_cols,
                model            = model,
                scaler_x         = scaler_x,
                scaler_y         = scaler_y,
                target_y_cols    = target_y_cols,
                y_cols           = y_cols,
            )

            st.markdown("### Simulation Results")
            for ty in target_y_cols:
                fig = px.line(
                    results_df,
                    x=vary_feat,
                    y=f"Predicted {ty}",
                    title=f"{vary_feat} → {ty}",
                )
                fig.update_layout(yaxis=dict(autorange=True))
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(results_df, use_container_width=True)
            csv_data = results_df.to_csv(index=False).encode("utf-8")

        # ---- Multi-feature sweep ----
        else:
            st.markdown("### Simulation Results (Per-Feature Sweep)")
            all_results = run_multi_sweep(
                sweep_arrays     = sweep_arrays,
                constant_features= constant_features,
                varying_features = varying_features,
                x_cols           = x_cols,
                df               = df,
                model            = model,
                scaler_x         = scaler_x,
                scaler_y         = scaler_y,
                target_y_cols    = target_y_cols,
                y_cols           = y_cols,
            )

            for r in all_results:
                vary_feat = r["x"]
                ty        = r["y"]
                res_df    = r["df"]
                fig = px.line(
                    res_df,
                    x=vary_feat,
                    y=f"Predicted {ty}",
                    title=f"{vary_feat} → {ty}",
                )
                fig.update_layout(yaxis=dict(autorange=True))
                st.plotly_chart(fig, use_container_width=True)

            # Combine all results into one downloadable table
            combined = pd.concat(
                [
                    r["df"].assign(**{"Varied X": r["x"], "Target Y": r["y"]})
                    for r in all_results
                ],
                ignore_index=True,
            )
            st.dataframe(combined, use_container_width=True)
            csv_data = combined.to_csv(index=False).encode("utf-8")

        # ---- Download button ----
        st.download_button(
            label="📥 Download Simulation Results (CSV)",
            data=csv_data,
            file_name="WhatIf_Simulation_Results.csv",
            mime="text/csv",
        )

        # ---- Save to sim history ----
        const_dict = {k: v["value"] for k, v in constant_features.items()}
        vary_dict  = {k: v["step_size"] for k, v in varying_features.items()}
        st.session_state.sim_history.append(
            {
                "Timestamp":        datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Varying Features": ", ".join(varying_features.keys()),
                "Target KPIs":      ", ".join(target_y_cols),
                "Step Sizes":       str(vary_dict),
                "constants":        const_dict,
            }
        )
        st.success("✅ Simulation Run Saved to Action History!")

    # ------------------------------------------------------------------ #
    # Simulation action history
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.markdown("### 🕒 Simulation Action History")

    if not st.session_state.sim_history:
        st.info("No actions performed yet. Run a simulation to save it to history.")
    else:
        history_df = pd.DataFrame(st.session_state.sim_history).drop(
            columns=["constants"]
        )
        st.dataframe(history_df, use_container_width=True)

        st.markdown("**Load a Past Action Scenario:**")
        selected_timestamp = st.selectbox(
            "Select Action by Timestamp",
            [h["Timestamp"] for h in reversed(st.session_state.sim_history)],
        )
        if st.button("Load Selected Scenario"):
            scenario = next(
                h
                for h in st.session_state.sim_history
                if h["Timestamp"] == selected_timestamp
            )
            st.session_state.loaded_sim = scenario
            st.success(
                f"Scenario from {selected_timestamp} loaded! "
                "The constant feature inputs have been updated."
            )
            st.rerun()
