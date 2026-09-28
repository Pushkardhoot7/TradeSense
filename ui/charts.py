"""
Charts Module for TradeSense UI

Provides Plotly visualization functions for price series, correlation heatmaps, NetworkX graph plots, Hasse diagrams, Venn diagrams, and color clusters.
Enforces a consistent Plotly template (plotly_white) across all tabs.
"""

from typing import Dict, Any, List, Set
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Shared Plotly Light/Neutral Theme Template
SHARED_PLOTLY_TEMPLATE = "plotly_white"
PAPER_BG = "rgba(255, 255, 255, 0.0)"
PLOT_BG = "rgba(248, 250, 252, 0.8)"


def plot_stock_prices(price_df: pd.DataFrame) -> go.Figure:
    """Generate normalized stock price time-series chart (Base 100)."""
    if price_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Stock Price Data Available", template=SHARED_PLOTLY_TEMPLATE)
        return fig

    normalized = price_df.div(price_df.iloc[0]).mul(100)

    fig = px.line(
        normalized,
        x=normalized.index,
        y=normalized.columns,
        title="Rebased Historical Stock Performance (Base = 100)",
        labels={"value": "Normalized Index (Base 100)", "Date": "Date", "variable": "Stock Ticker"},
    )
    fig.update_layout(
        template=SHARED_PLOTLY_TEMPLATE,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame, method_name: str = "Pearson") -> go.Figure:
    """
    Generate interactive Plotly correlation heatmap with diverging color scale.
    """
    if corr_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Correlation Data Available", template=SHARED_PLOTLY_TEMPLATE)
        return fig

    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title=f"Stock Return {method_name} Correlation Matrix A (A = Aᵀ)",
        zmin=-1.0,
        zmax=1.0,
        labels=dict(x="Stock Symbol", y="Stock Symbol", color="Correlation (r)"),
    )
    fig.update_traces(
        hovertemplate="<b>Stock i:</b> %{y}<br><b>Stock j:</b> %{x}<br><b>Pearson Correlation A[i,j]:</b> %{z:.4f}<extra></extra>"
    )
    fig.update_layout(
        template=SHARED_PLOTLY_TEMPLATE,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_network_graph(G: nx.Graph, threshold: float = 0.5, coloring: Dict[str, int] = None) -> go.Figure:
    """Render 2D interactive NetworkX stock correlation network graph via Plotly."""
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="No Network Nodes Available", template=SHARED_PLOTLY_TEMPLATE)
        return fig

    pos = nx.spring_layout(G, seed=42)

    # Edge Traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="#0284c7"),
        hoverinfo="none",
        mode="lines",
    )

    # Node Traces
    node_x, node_y, node_text, node_colors = [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = G.degree(node)
        color_val = coloring.get(node, 0) if coloring else degree
        node_colors.append(color_val)
        node_text.append(f"<b>{node}</b><br/>Degree: {degree}<br/>Color Cluster: {color_val}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[node for node in G.nodes()],
        textposition="top center",
        textfont=dict(color="#0f172a", size=11, family="Inter"),
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=node_colors,
            size=24,
            colorbar=dict(
                thickness=15,
                title=dict(text="Color Cluster" if coloring else "Node Degree", side="right"),
                xanchor="left",
            ),
            line=dict(width=2, color="#ffffff"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Stock Correlation Topology Graph (Correlation ≥ {threshold})",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template=SHARED_PLOTLY_TEMPLATE,
            paper_bgcolor=PAPER_BG,
            plot_bgcolor=PLOT_BG,
        ),
    )
    return fig


def plot_hasse_diagram(hasse_dag: nx.DiGraph) -> go.Figure:
    """Render interactive top-down Hasse Diagram DAG using Plotly."""
    if hasse_dag.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="No Dominance Relations Found (Empty Hasse Diagram)", template=SHARED_PLOTLY_TEMPLATE)
        return fig

    try:
        for n in hasse_dag.nodes():
            hasse_dag.nodes[n]["subset"] = hasse_dag.in_degree(n)
        pos = nx.multipartite_layout(hasse_dag, subset_key="subset")
    except Exception:
        pos = nx.spring_layout(hasse_dag, seed=42)

    edge_x, edge_y = [], []
    for edge in hasse_dag.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color="#6366f1"),
        hoverinfo="none",
        mode="lines",
    )

    node_x, node_y, node_text = [], [], []
    for node in hasse_dag.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        in_deg = hasse_dag.in_degree(node)
        out_deg = hasse_dag.out_degree(node)
        status = "Maximal (Undominated)" if in_deg == 0 else ("Minimal (Dominated)" if out_deg == 0 else "Intermediate")
        node_text.append(f"<b>{node}</b><br/>Status: {status}<br/>Dominates: {out_deg} stocks<br/>Dominated by: {in_deg} stocks")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[node for node in hasse_dag.nodes()],
        textposition="top center",
        textfont=dict(color="#0f172a", size=12, family="Inter"),
        hovertext=node_text,
        marker=dict(
            size=26,
            color="#0284c7",
            line=dict(width=2, color="#ffffff"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Pareto Stock Dominance Hasse Diagram (Transitive Reduction DAG)",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template=SHARED_PLOTLY_TEMPLATE,
            paper_bgcolor=PAPER_BG,
            plot_bgcolor=PLOT_BG,
        ),
    )
    return fig


def plot_venn_diagram(
    set_a_name: str, set_b_name: str, set_a: Set[str], set_b: Set[str]
) -> go.Figure:
    """
    Render a 2-set Venn diagram visualization using Plotly shapes and scatter annotations.
    """
    only_a = set_a - set_b
    only_b = set_b - set_a
    both = set_a.intersection(set_b)

    fig = go.Figure()

    # Draw two overlapping circles as shapes
    fig.add_shape(
        type="circle",
        x0=0.0, y0=0.0, x1=2.0, y1=2.0,
        fillcolor="rgba(56, 189, 248, 0.3)",
        line_color="#0284c7",
        line_width=2,
    )
    fig.add_shape(
        type="circle",
        x0=1.0, y0=0.0, x1=3.0, y1=2.0,
        fillcolor="rgba(168, 85, 247, 0.3)",
        line_color="#7e22ce",
        line_width=2,
    )

    # Annotations for Region Labels and Stock Lists
    annotations = [
        # Set A Label & Count
        dict(x=0.5, y=1.7, text=f"<b>{set_a_name}</b><br/>({len(set_a)} stocks)", showarrow=False, font=dict(size=14, color="#0369a1")),
        dict(x=0.5, y=1.0, text=f"<b>Only A ({len(only_a)}):</b><br/>" + "<br/>".join(sorted(list(only_a))[:5]), showarrow=False, font=dict(size=11, color="#334155")),
        
        # Set B Label & Count
        dict(x=2.5, y=1.7, text=f"<b>{set_b_name}</b><br/>({len(set_b)} stocks)", showarrow=False, font=dict(size=14, color="#6b21a8")),
        dict(x=2.5, y=1.0, text=f"<b>Only B ({len(only_b)}):</b><br/>" + "<br/>".join(sorted(list(only_b))[:5]), showarrow=False, font=dict(size=11, color="#334155")),
        
        # Intersection Label & Count
        dict(x=1.5, y=1.3, text=f"<b>A ∩ B ({len(both)}):</b><br/>" + "<br/>".join(sorted(list(both))[:5]), showarrow=False, font=dict(size=12, color="#0f172a")),
    ]

    fig.update_layout(
        title=f"Venn Diagram Set Operation: {set_a_name} vs {set_b_name}",
        annotations=annotations,
        xaxis=dict(range=[-0.2, 3.2], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.2, 2.2], showgrid=False, zeroline=False, showticklabels=False),
        template=SHARED_PLOTLY_TEMPLATE,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
