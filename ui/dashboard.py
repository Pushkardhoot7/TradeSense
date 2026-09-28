"""
Top-Level Dashboard Integration & Presentation Layer for TradeSense

Features:
- Exact Branding: TradeSense | "Intelligent Market Insights"
- Left Sidebar Navigation Menu: Dashboard, Stock Analysis, Correlation Network, Relations & Hasse, Graph Coloring, Portfolio Optimizer, Reports
- Top Header Metadata & Status Badges
- 6 KPI Cards, Stock Trend Table, Correlation Heatmap & Network Graph, Poset Hasse DAG, Welsh-Powell Coloring, Set Theory & Boolean Screening, Combinatorial Portfolio Optimizer, Top 3 Portfolio Cards, Final Mathematical Summary & Viva Q&A Reports Page
- Explicit Data Error Handling with [Switch to Demo Mode] option
"""

from datetime import datetime
import pandas as pd
import streamlit as st

from modules.stock_selector import get_sectors
from modules.safe_pipeline import execute_safe_pipeline
from ui.styles import apply_custom_styles
from ui.charts import plot_stock_prices, plot_correlation_heatmap, plot_network_graph, plot_hasse_diagram, plot_venn_diagram
from ui.display_helpers import render_top_header, render_data_source_error, render_pipeline_flow_diagram, render_explanation_box


def render_dashboard():
    """Render complete TradeSense financial analytics dashboard."""
    apply_custom_styles()

    # Handle demo mode override session state
    if "demo_mode_override" not in st.session_state:
        st.session_state["demo_mode_override"] = True

    # LEFT SIDEBAR NAVIGATION & CONTROLS
    st.sidebar.markdown("### 📈 TradeSense Navigation")
    nav_choice = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard Home",
            "Stock Analysis",
            "Correlation Network",
            "Relations & Hasse",
            "Graph Coloring & Screening",
            "Portfolio Optimizer",
            "Reports & Project Guide",
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ ANALYTICS INPUT PANEL")

    market_choice = st.sidebar.selectbox("Market Exchange", ["NSE"])
    all_sectors = get_sectors()
    sector_choice = st.sidebar.selectbox("Select Sector", ["All Sectors"] + all_sectors)
    st.sidebar.caption("Selection Rule: Top 15 Per Sector")

    period_display_map = {
        "3 Months": "3m",
        "6 Months": "6m",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
    }
    selected_period_label = st.sidebar.selectbox("Historical Period", list(period_display_map.keys()), index=2)
    selected_period_code = period_display_map[selected_period_label]

    interval_choice = st.sidebar.selectbox("Data Interval", ["Daily"])
    interval_code = "1d"

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 MATHEMATICAL THRESHOLDS")

    corr_threshold = st.sidebar.slider("Correlation Threshold (|r| ≥)", 0.0, 1.0, 0.70, 0.05)
    return_threshold = st.sidebar.slider("Return Threshold (%)", 0.0, 50.0, 5.0, 1.0)
    risk_threshold = st.sidebar.slider("Risk Threshold (%)", 5.0, 50.0, 20.0, 1.0)
    portfolio_k = st.sidebar.number_input("Portfolio Size (k Stocks)", min_value=2, max_value=6, value=3, step=1)

    analyze_pressed = st.sidebar.button("🚀 ANALYZE MARKET", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ DATA STATUS & SETTINGS")

    is_demo_mode = st.sidebar.toggle(
        "Demo Mode (Offline Sample Data)",
        value=st.session_state["demo_mode_override"],
        help="Default ON for presentation safety. Disabling attempts live download from Yahoo Finance."
    )
    st.session_state["demo_mode_override"] = is_demo_mode

    if is_demo_mode:
        st.sidebar.markdown('<span class="badge-demo"><span>⚠️</span> DEMO / SAMPLE DATA</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="badge-live"><span>🌐</span> HISTORICAL / YAHOO FINANCE</span>', unsafe_allow_html=True)

    # RUN PIPELINE ON BUTTON CLICK OR INITIAL LOAD
    inputs = {
        "market": market_choice,
        "sector": sector_choice,
        "period_label": selected_period_label,
        "period_code": selected_period_code,
        "interval_code": interval_code,
        "corr_threshold": corr_threshold,
        "return_threshold": return_threshold,
        "risk_threshold": risk_threshold,
        "portfolio_k": int(portfolio_k),
    }

    if analyze_pressed or "pipeline_results" not in st.session_state:
        with st.spinner("Executing Discrete Mathematics Market Pipeline..."):
            res = execute_safe_pipeline(inputs, is_demo_mode=is_demo_mode)
            st.session_state["pipeline_results"] = res

    results = st.session_state.get("pipeline_results", {})

    # HANDLE DATA RETRIEVAL ERRORS
    if "error" in results:
        render_top_header(results=None, is_demo_mode=is_demo_mode)
        render_data_source_error(results["error"])
        return

    # TOP HEADER DISPLAY
    render_top_header(results=results, is_demo_mode=is_demo_mode)

    # PAGE ROUTING IMPLEMENTATION

    # PAGE 1: DASHBOARD HOME
    if nav_choice == "Dashboard Home":
        st.markdown("## 📊 Executive Market Dashboard")
        st.caption("Real-time discrete mathematical summary of historical stock co-movement and dominance rankings.")

        # Top 6 KPI Cards (Single Row)
        kpi_cols = st.columns(6)
        with kpi_cols[0]:
            st.metric("TOTAL STOCKS ANALYZED", len(results["selected_tickers"]))
        with kpi_cols[1]:
            st.metric("STRONG RELATIONSHIPS", results["g_stats"]["num_edges"])
        with kpi_cols[2]:
            st.metric("NON-DOMINATED STOCKS", len(results["non_dominated_stocks"]))
        with kpi_cols[3]:
            st.metric("GRAPH COLORS", len(results["color_groups"]))
        with kpi_cols[4]:
            st.metric("PORTFOLIOS EVALUATED", results["evaluated_count"])
        with kpi_cols[5]:
            st.metric("TOP DM SCORE", f"{results['top_dm_score']:.2f}")

        st.markdown("---")

        # Section 1: Stock Trend Overview
        st.markdown("### 📋 Stock Overview & Trend Performance")
        df_overview = results["df_metrics"].copy()

        # Add Trend Badges
        def compute_trend_badge(ret):
            if ret > 10.0:
                return "Positive 🟢"
            elif ret < -5.0:
                return "Negative 🔴"
            else:
                return "Neutral 🟡"

        df_overview["Trend"] = df_overview["Return %"].apply(compute_trend_badge)
        df_overview["Rank #"] = range(1, len(df_overview) + 1)

        display_cols = ["Rank #", "Stock", "Sector", "Latest Price", "Return %", "Risk %", "Average Volume", "Trend"]
        avail_cols = [c for c in display_cols if c in df_overview.columns]

        st.dataframe(
            df_overview[avail_cols].style.format({
                "Latest Price": "₹{:.2f}",
                "Return %": "{:+.2f}%",
                "Risk %": "{:.2f}%",
                "Average Volume": "{:,.0f}"
            }),
            use_container_width=True
        )

        st.markdown("---")
        render_pipeline_flow_diagram()

    # PAGE 2: STOCK ANALYSIS
    elif nav_choice == "Stock Analysis":
        st.markdown("## 🔍 Detailed Stock Analysis & Price Trends")
        st.caption("Inspect individual historical price series, return distributions, and annualized risk metrics.")

        selected_stock = st.selectbox("Select Stock for Deep Dive", results["selected_tickers"])
        if selected_stock in results["price_df"].columns:
            fig_price = plot_stock_prices(results["price_df"][[selected_stock]])
            st.plotly_chart(fig_price, use_container_width=True)

        st.markdown("#### 📋 Universe Stock Metrics Table")
        st.dataframe(results["df_metrics"], use_container_width=True)

    # PAGE 3: CORRELATION NETWORK
    elif nav_choice == "Correlation Network":
        st.markdown("## 🔗 Pearson Correlation Matrix & Relationship Graph G=(V,E)")

        # Section 2: Correlation Matrix
        st.markdown("### 1️⃣ Correlation Matrix")
        st.caption("Pairwise historical correlation of daily returns")

        c_stat1, c_stat2, c_stat3 = st.columns(3)
        with c_stat1:
            st.metric("Strongest Positive Pair", f"{results['corr_stats']['max_pair'][0]} - {results['corr_stats']['max_pair'][1]}", f"r = {results['corr_stats']['max_corr']:.2f}")
        with c_stat2:
            st.metric("Weakest Pair", f"{results['corr_stats']['min_pair'][0]} - {results['corr_stats']['min_pair'][1]}", f"r = {results['corr_stats']['min_corr']:.2f}")
        with c_stat3:
            st.metric("Average Correlation", f"{results['corr_stats']['mean_corr']:.2f}")

        fig_corr = plot_correlation_heatmap(results["corr_df"])
        st.plotly_chart(fig_corr, use_container_width=True)

        render_explanation_box(
            title="Correlation Matrix",
            plain_english_meaning="Each cell represents the historical correlation between two stocks. Values close to +1 indicate similar historical movement, while values close to 0 indicate weaker linear relationship."
        )

        st.markdown("---")

        # Section 3: Correlation Network Graph
        st.markdown("### 2️⃣ Correlation Network Graph G=(V,E)")
        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        with col_g1:
            st.metric("Nodes (|V|)", results["g_stats"]["num_vertices"])
        with col_g2:
            st.metric("Edges (|E|)", results["g_stats"]["num_edges"])
        with col_g3:
            st.metric("Average Degree", f"{results['g_stats']['avg_degree']:.2f}")
        with col_g4:
            hd = results["g_stats"]["highest_degree_stock"]
            st.metric("Most Connected Stock", hd["stock"], f"Degree {hd['degree']}")

        st.markdown("##### 🏷️ Edge Legend: **Strong** (r ≥ 0.85) | **Moderate** (0.70 ≤ r < 0.85) | **Weak** (r < 0.70)")

        fig_net = plot_network_graph(results["G"], threshold=results["inputs"]["corr_threshold"], coloring=results["coloring_dict"])
        st.plotly_chart(fig_net, use_container_width=True)

        render_explanation_box(
            title="Correlation Network Graph",
            plain_english_meaning="Vertices (nodes) represent stocks, and edges connect pairs with historical correlation exceeding the threshold. Highly connected nodes represent centralized market risk factors."
        )

    # PAGE 4: RELATIONS & HASSE
    elif nav_choice == "Relations & Hasse":
        st.markdown("## 🔺 Return-Risk Dominance Relation & Poset Hasse Diagram")

        # Section 5: Dominance Relation
        st.markdown("### 1️⃣ Return-Risk Dominance Relation (A ≽ B)")
        st.caption("Definition: Stock A ≽ Stock B iff Return(A) ≥ Return(B) AND Risk(A) ≤ Risk(B)")

        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            st.success(f"REFLEXIVE **{'✓' if results['is_refl'] else '✗'}**")
        with col_p2:
            st.success(f"ANTISYMMETRIC **{'✓' if 'Antisymmetric' in results['antisym_info']['message'] else '✗'}**")
        with col_p3:
            st.success(f"TRANSITIVE **{'✓' if results['is_trans'] else '✗'}**")
        with col_p4:
            st.info(f"PARTIAL ORDER: **{'YES' if results['is_poset'] else 'NO'}**")

        st.markdown("#### 📋 Concrete Dominance Pair Examples (A ≽ B)")
        if results["dominance_pairs"]:
            dom_df = pd.DataFrame(results["dominance_pairs"], columns=["Dominant Stock (A)", "Dominated Stock (B)"])
            st.dataframe(dom_df.head(10), use_container_width=True)
        else:
            st.write("No strict dominance pairs found.")

        st.markdown("---")

        # Section 6: Hasse Diagram
        st.markdown("### 2️⃣ Hasse Diagram (Transitive Reduction Cover DAG)")
        fig_hasse = plot_hasse_diagram(results["hasse_dag"])
        st.plotly_chart(fig_hasse, use_container_width=True)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("#### 🏆 Maximal Elements (Undominated Leaders)")
            st.write(", ".join(results["maximal_elems"]) if results["maximal_elems"] else "None")
        with col_m2:
            st.markdown("#### 🔻 Minimal Elements (Dominated Assets)")
            st.write(", ".join(results["minimal_elems"]) if results["minimal_elems"] else "None")
        with col_m3:
            st.markdown("#### ⚖️ Incomparable Pairs Count")
            st.write(f"Count: **{len(results['incomparable_pairs'])}**")

        render_explanation_box(
            title="Hasse Diagram & Pareto Dominance",
            plain_english_meaning="The Hasse diagram represents the partial-order structure created by the defined return-risk dominance relation. Maximal elements at the top represent undominated Pareto-optimal stock choices."
        )

    # PAGE 5: GRAPH COLORING & SCREENING
    elif nav_choice == "Graph Coloring & Screening":
        st.markdown("## 🎨 Welsh-Powell Graph Coloring & Set/Boolean Screening")

        # Section 7: Graph Coloring
        st.markdown("### 1️⃣ Graph Coloring (Independent Stock Clusters)")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.metric("Colors Used", len(results["color_groups"]))
        with col_c2:
            st.metric("Coloring Validity", "VALID ✓" if results["is_coloring_valid"] else "INVALID ✗")
        with col_c3:
            if results["chromatic_info"]["exact"]:
                st.metric("Verified Chromatic Number χ(G)", results["chromatic_info"]["chromatic_number"])
            else:
                st.metric("Chromatic Upper Bound", f"≤ {len(results['color_groups'])}")

        fig_net_col = plot_network_graph(results["G"], threshold=results["inputs"]["corr_threshold"], coloring=results["coloring_dict"])
        st.plotly_chart(fig_net_col, use_container_width=True)

        st.markdown("#### 📋 Stock Color Groups (Independent Sets)")
        color_rows = []
        for c_id, stocks_in_cluster in results["color_groups"].items():
            color_rows.append({
                "Group ID": f"GROUP {c_id + 1}",
                "Stock Count": len(stocks_in_cluster),
                "Stocks in Independent Set": ", ".join(stocks_in_cluster)
            })
        st.dataframe(pd.DataFrame(color_rows), use_container_width=True)

        render_explanation_box(
            title="Graph Coloring",
            plain_english_meaning="Graph coloring partitions the relationship graph into mathematically valid independent sets. Adjacent nodes connected by high correlation MUST have different colors, ensuring stocks in the same color group are uncorrelated."
        )

        st.markdown("---")

        # Section 4 & 8: Set Theory & Boolean Logic
        st.markdown("### 2️⃣ Set Theory & Mathematical Screening Rule")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(f"#### Set A = High Return ({len(results['set_high_return'])})")
            st.caption(", ".join(results['set_high_return']) if results['set_high_return'] else "Empty")
        with col_s2:
            st.markdown(f"#### Set B = High Volume ({len(results['set_high_volume'])})")
            st.caption(", ".join(results['set_high_volume']) if results['set_high_volume'] else "Empty")
        with col_s3:
            st.markdown(f"#### Set C = Low Risk ({len(results['set_low_risk'])})")
            st.caption(", ".join(results['set_low_risk']) if results['set_low_risk'] else "Empty")

        fig_venn = plot_venn_diagram(
            "High Return", "Low Risk", results["set_high_return"], results["set_low_risk"]
        )
        st.plotly_chart(fig_venn, use_container_width=True)

        st.markdown("#### ⚡ Boolean Mathematical Screening Table")
        st.caption("Rule: QUALIFIED = (Return > 5%) AND (Risk < 20%) AND (Volume > Average Volume)")
        st.dataframe(
            results["df_boolean_screening"].style.map(
                lambda v: "color: #16a34a; font-weight: bold;" if v is True else "color: #dc2626;",
                subset=["Return Condition", "Risk Condition", "Volume Condition", "Final Result"]
            ),
            use_container_width=True
        )

    # PAGE 6: PORTFOLIO OPTIMIZER
    elif nav_choice == "Portfolio Optimizer":
        st.markdown("## 💼 Portfolio Optimizer & Top 3 Portfolio Insights")

        # Section 9: Combinatorics
        st.markdown("### 1️⃣ Combinatorial Portfolio Generation")
        col_pa1, col_pa2, col_pa3, col_pa4 = st.columns(4)
        with col_pa1:
            st.metric("Eligible Stocks", results["eligible_stocks_count"])
        with col_pa2:
            st.metric("Candidate Pool (N)", len(results["candidate_pool"]))
        with col_pa3:
            st.metric(f"Possible Combinations C({len(results['candidate_pool'])},{results['inputs']['portfolio_k']})", results["possible_combinations"])
        with col_pa4:
            st.metric("Evaluated Combinations", results["evaluated_count"])

        st.markdown("---")

        # Section 10 & 11: Top 3 Portfolios & DM Score
        st.markdown("### 2️⃣ TOP 3 PORTFOLIO INSIGHTS")

        top3 = results["top3_structured_data"]
        medals = ["🥇 RANK #1", "🥈 RANK #2", "🥉 RANK #3"]
        styles = ["rank-card-1", "rank-card-2", "rank-card-3"]

        for idx, item in enumerate(top3):
            st.markdown(
                f"""
                <div class="rank-card {styles[idx]}">
                    <div style="font-size: 1.25rem; font-weight: 800; color: #1e1b4b; margin-bottom: 0.5rem;">
                        {medals[idx]}: {", ".join(item['portfolio'])}
                    </div>
                    <div style="display: flex; gap: 2rem; margin-bottom: 1rem;">
                        <div><span style="color: #64748b; font-size: 0.85rem;">DM SCORE</span><br><b style="font-size: 1.5rem; color: #4f46e5;">{item['dm_score']:.2f} / 100</b></div>
                        <div><span style="color: #64748b; font-size: 0.85rem;">HISTORICAL RETURN</span><br><b style="font-size: 1.2rem; color: #16a34a;">+{item['return_raw']:.2f}%</b></div>
                        <div><span style="color: #64748b; font-size: 0.85rem;">PORTFOLIO RISK</span><br><b style="font-size: 1.2rem; color: #dc2626;">{item['risk_raw']:.2f}%</b></div>
                        <div><span style="color: #64748b; font-size: 0.85rem;">AVG CORRELATION</span><br><b style="font-size: 1.2rem; color: #0284c7;">{item['avg_correlation']:.2f}</b></div>
                    </div>
                    <div style="font-size: 0.85rem; color: #475569;">
                        <b>Graph Color Groups:</b> {item['color_groups_count']} distinct clusters | <b>Pareto Dominance:</b> {item['dominance_status']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        render_explanation_box(
            title="HOW IS DM SCORE CALCULATED?",
            plain_english_meaning="DM Score evaluates portfolios on a 0–100 scale using five weighted metrics: Return Score (30%), Covariance Risk Score (25%), Correlation Diversification Score (25%), Graph Diversity Score (10%), and Pareto Dominance Score (10%). Candidates are ranked deterministically."
        )

    # PAGE 7: REPORTS & PROJECT GUIDE
    elif nav_choice == "Reports & Project Guide":
        st.markdown("## 📑 Project Technical Reports & Mathematical Guide")

        # Section 12: Final Mathematical Summary
        st.markdown("### 1️⃣ Final Mathematical Run Summary")

        sum_cols = st.columns(3)
        with sum_cols[0]:
            st.write(f"• **Stocks Analyzed:** {len(results['selected_tickers'])}")
            st.write(f"• **Correlation Threshold:** |r| ≥ {results['inputs']['corr_threshold']:.2f}")
            st.write(f"• **Strong Relationships:** {results['g_stats']['num_edges']}")
        with sum_cols[1]:
            st.write(f"• **Dominance Pairs:** {len(results['dominance_pairs'])}")
            st.write(f"• **Partial Order Poset:** {'YES ✓' if results['is_poset'] else 'NO ✗'}")
            st.write(f"• **Colors Used:** {len(results['color_groups'])}")
        with sum_cols[2]:
            st.write(f"• **Candidate Pool (N):** {len(results['candidate_pool'])}")
            st.write(f"• **Portfolios Evaluated:** {results['evaluated_count']}")
            st.write(f"• **Top DM Score:** {results['top_dm_score']:.2f}")

        st.markdown("---")
        render_pipeline_flow_diagram()

        st.markdown("---")
        st.markdown("### 2️⃣ Discrete Mathematics Concept Reference Guide")

        concepts = [
            ("Pearson Correlation Matrix", "Measures linear co-movement A[i,j] between daily stock returns on [-1, +1].", "A = Aᵀ, diagonal=1.0"),
            ("Correlation Graph G=(V,E)", "Network graph linking stocks with correlation exceeding threshold.", "Edge (u,v) iff |r(u,v)| ≥ threshold"),
            ("Dominance Binary Relation (≽)", "Partial order relation where Stock A dominates B iff Return(A) ≥ Return(B) AND Risk(A) ≤ Risk(B).", "Return(A) ≥ Return(B) ∧ Risk(A) ≤ Risk(B)"),
            ("Hasse Diagram", "Visual cover relation DAG formed by removing self-loops and transitive shortcuts.", "Transitive reduction cover DAG"),
            ("Welsh-Powell Graph Coloring", "Assigns colors to nodes so no two adjacent correlated nodes share a color.", "Independent sets of non-correlated assets"),
            ("Set Theory Partitions", "Groups stocks by mathematical criteria (High Return, Low Risk, High Volume).", "Set intersections S_R ∩ S_V ∩ S_Vol"),
            ("Boolean Logic Screening", "Evaluates boolean truth table rules to identify qualified stocks.", "QUALIFIED = (Return > 5%) ∧ (Risk < 20%)"),
            ("Combinatorics C(N,k)", "Calculates combinations choosing k stocks from candidate pool N.", "C(N,k) = N! / (k!(N-k)!)"),
            ("DM Portfolio Score", "Composite 5-component weighted evaluation metric (0–100 scale).", "0.30R + 0.25Risk + 0.25Div + 0.10Group + 0.10Dom"),
        ]

        for title, desc, formula in concepts:
            with st.expander(f"📖 {title}", expanded=False):
                st.markdown(f"- **Explanation:** {desc}")
                st.markdown(f"- **Mathematical Rule:** `{formula}`")
