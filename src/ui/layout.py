"""
src/ui/layout.py
================
Page-level Streamlit configuration and sidebar navigation.

Call ``configure_page()`` once at the very start of app.py (before any
other ``st.*`` call) to set page config and inject the CSS theme.

Call ``render_sidebar()`` to render the navigation menu and get the
currently selected page name.
"""
from __future__ import annotations

import streamlit as st
from streamlit_option_menu import option_menu

from config.settings import (
    NAVIGATION_ICONS,
    NAVIGATION_OPTIONS,
    PAGE_LAYOUT,
    PAGE_TITLE,
    SIDEBAR_STATE,
    THEME_CSS,
)


def configure_page() -> None:
    """
    Set Streamlit page configuration and inject the premium CSS theme.

    Must be the first ``st.*`` call in app.py.
    """
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout=PAGE_LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE,
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_sidebar() -> str:
    """
    Render the sidebar header and option-menu navigation.

    Returns
    -------
    The label of the currently selected page (e.g. "Overview").
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 1.2rem 0.5rem 0.4rem 0.5rem;">
              <div style="font-size:10px; letter-spacing:0.18em; color:rgba(255,255,255,0.65);
                          font-family:'Inter',sans-serif; text-transform:uppercase;
                          margin-bottom:4px;">Soft Sensor Platform</div>
              <div style="font-family:'Outfit',sans-serif; font-weight:800;
                          font-size:1.6rem; line-height:1.1;
                          background: linear-gradient(90deg, #ffffff 0%, #a8d4ff 100%);
                          -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                          background-clip: text;">
                Multi X-Y
              </div>
              <div style="font-size:11px; color:rgba(255,255,255,0.75); font-family:'Inter',sans-serif;
                          margin-top:2px; font-weight:500;">Industrial DAE &middot; ML Dashboard</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="height:1px; margin: 0.3rem 0 0.8rem 0;
                        background: linear-gradient(90deg,
                          transparent 0%, rgba(255,255,255,0.35) 40%,
                          rgba(168,212,255,0.25) 70%, transparent 100%);"></div>
            """,
            unsafe_allow_html=True,
        )

        selected = option_menu(
            menu_title=None,
            options=NAVIGATION_OPTIONS,
            icons=NAVIGATION_ICONS,
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "0 !important",
                    "background-color": "#0f2a52 !important",
                    "border-radius": "0 !important",
                    "border": "none !important",
                    "box-shadow": "none !important",
                },
                "icon": {
                    "color": "rgba(255,255,255,0.70)",
                    "font-size": "17px",
                },
                "nav-link": {
                    "font-size": "14.5px",
                    "font-weight": "400",
                    "font-family": "'Inter', sans-serif",
                    "color": "rgba(255,255,255,0.80)",
                    "text-align": "left",
                    "margin": "1px 0",
                    "padding": "0.45rem 0.75rem",
                    "border-radius": "8px",
                    "--hover-color": "rgba(255,255,255,0.12)",
                },
                "nav-link-selected": {
                    "background": "rgba(255,255,255,0.15)",
                    "border-left": "3px solid #a8d4ff",
                    "color": "#ffffff",
                    "font-weight": "700",
                    "border-radius": "0 8px 8px 0",
                },
            },
        )

        st.markdown(
            """
            <div style="margin-top: 2rem; padding: 0.5rem 0.5rem 0;
                        border-top: 1px solid rgba(255,255,255,0.05);">
              <div style="font-size:9.5px; color:rgba(255,255,255,0.40); font-family:'Inter',sans-serif;
                          letter-spacing:0.04em;">
                v1.0 &nbsp;&middot;&nbsp; PyTorch DAE &nbsp;&middot;&nbsp; Streamlit
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected
