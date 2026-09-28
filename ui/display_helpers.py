"""
Display Helpers and Adapter Functions for TradeSense UI

Provides header rendering, data source error blocks, academic callout boxes, and the 9-stage Project Pipeline flow diagram.
"""

from datetime import datetime
import streamlit as st


def render_top_header(results: dict = None, is_demo_mode: bool = True):
    """
    Render exact brand header:
    Title: TradeSense
    Subtitle: "Intelligent Market Insights"
    Secondary Subtitle: "Discrete Mathematics Based Stock Market Analysis & Portfolio Insights"
    """
    last_updated = results.get("fetch_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")) if results else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    period_label = results.get("inputs", {}).get("period_label", "1 Year") if results else "1 Year"

    mode_badge_html = (
        '<span class="badge-demo"><span>⚠️</span> DEMO MODE / SAMPLE DATA</span>'
        if is_demo_mode
        else '<span class="badge-live"><span>🌐</span> HISTORICAL DATA / YAHOO FINANCE</span>'
    )

    st.markdown(
        f"""
        <div class="top-header-container">
            <div>
                <div class="brand-title">
                    <span>📈</span> TradeSense
                </div>
                <div class="brand-subtitle">Intelligent Market Insights</div>
                <div class="brand-sec-subtitle">Discrete Mathematics Based Stock Market Analysis & Portfolio Insights</div>
            </div>
            <div style="text-align: right;">
                <div style="margin-bottom: 0.3rem;">{mode_badge_html}</div>
                <div style="font-size: 0.85rem; color: #475569;"><b>Market:</b> NSE | <b>Period:</b> {period_label}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;"><b>Last Updated:</b> {last_updated}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_source_error(error_msg: str):
    """
    Render explicit DATA SOURCE ERROR card when live market data fails:
    DATA SOURCE ERROR
    Reason: <actual error>
    [Switch to Demo Mode]
    """
    st.markdown(
        f"""
        <div class="data-error-card">
            <h3 style="margin-top: 0; color: #991b1b; font-size: 1.25rem;">❌ DATA SOURCE ERROR</h3>
            <p><b>Reason:</b> {error_msg}</p>
            <p style="font-size: 0.9rem; color: #7f1d1d;">Unable to retrieve live market data for selected tickers from Yahoo Finance.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("🔄 Switch to Demo Mode", type="primary"):
        st.session_state["demo_mode_override"] = True
        st.rerun()


def render_pipeline_flow_diagram():
    """
    Render 9-stage connected pipeline flow diagram:
    MARKET DATA -> MATRIX -> GRAPH -> RELATION -> HASSE -> COLORING -> COMBINATORICS -> DM SCORE -> PORTFOLIO INSIGHTS
    """
    st.markdown("### 🔄 Project Mathematical Pipeline Flow")
    st.caption("Complete processing flow mapping quantitative market data to discrete mathematical structures.")

    pipeline_stages = [
        {"step": "Stage 1", "title": "MARKET DATA", "caption": "Loads historical daily price series from NSE universe."},
        {"step": "Stage 2", "title": "MATRIX", "caption": "Measures pairwise correlation (A = Aᵀ) between returns."},
        {"step": "Stage 3", "title": "GRAPH", "caption": "Builds relationship network G=(V,E) filtered by threshold."},
        {"step": "Stage 4", "title": "RELATION", "caption": "Evaluates Pareto dominance (A ≽ B) over return & risk."},
        {"step": "Stage 5", "title": "HASSE", "caption": "Applies transitive reduction to reveal cover DAG."},
        {"step": "Stage 6", "title": "COLORING", "caption": "Clusters non-correlated stocks into independent sets via Welsh-Powell."},
        {"step": "Stage 7", "title": "COMBINATORICS", "caption": "Generates candidate stock portfolios C(N, k) with pool limits."},
        {"step": "Stage 8", "title": "DM SCORE", "caption": "Evaluates candidates across 5 weighted return-risk-diversity metrics."},
        {"step": "Stage 9", "title": "PORTFOLIO INSIGHTS", "caption": "Selects Top 3 optimal portfolios with complete score breakdowns."},
    ]

    cols = st.columns(len(pipeline_stages))
    for idx, stage in enumerate(pipeline_stages):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="pipeline-node">
                    <div class="pipeline-step-num">{stage['step']}</div>
                    <div class="pipeline-node-title">{stage['title']}</div>
                    <div class="pipeline-node-caption">{stage['caption']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")


def render_explanation_box(title: str, plain_english_meaning: str, details: str = None):
    """
    Render standardized explanation block:
    [Section Title]
    "What does this mean?"
    One or two sentence plain-English explanation — no jargon.
    """
    st.markdown(
        f"""
        <div class="explanation-callout">
            <div class="explanation-title">💡 {title} — "What does this mean?"</div>
            <div>{plain_english_meaning}</div>
            {f'<div style="margin-top: 0.5rem; font-size: 0.85rem; color: #475569;"><b>Technical Details:</b> {details}</div>' if details else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )
