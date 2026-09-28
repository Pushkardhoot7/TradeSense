"""
TradeSense - Intelligent Market Insights

Discrete Mathematics Based Stock Market Analysis & Portfolio Insights
Main Streamlit Application Entry Point.

Module Dependencies:
- modules.stock_selector
- modules.data_loader
- modules.financial_metrics
- modules.correlation
- modules.graph_analysis
- modules.relations
- modules.hasse
- modules.coloring
- modules.set_theory
- modules.portfolio
- modules.dm_scoring
- modules.top3_display
- ui.dashboard
- ui.charts
- ui.display_helpers
"""

import streamlit as st
from ui.dashboard import render_dashboard

# Streamlit Page Configuration
st.set_page_config(
    page_title="TradeSense — Intelligent Market Insights",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main execution entry point."""
    render_dashboard()


if __name__ == "__main__":
    main()
