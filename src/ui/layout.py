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
            "<h2 style='text-align: left; margin-bottom: 0px;'>Multi X-Y</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h4 style='text-align: left; color: #4da6ff; margin-top: 0px;'>"
            "ML Dashboard</h4>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        selected = option_menu(
            menu_title=None,
            options=NAVIGATION_OPTIONS,
            icons=NAVIGATION_ICONS,
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "transparent",
                },
                "icon": {"color": "white", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#2d3748",
                },
                "nav-link-selected": {"background-color": "#2b6cb0"},
            },
        )

    return selected
