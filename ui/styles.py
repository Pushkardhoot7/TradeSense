"""
Styles Module for TradeSense UI

Provides clean, professional financial analytics dashboard CSS rules for Streamlit:
- Light/neutral background (#f8fafc)
- Dark readable typography (#0f172a)
- Indigo/Purple accent colors (#4f46e5)
- Green positive (#16a34a), Red negative (#dc2626), Amber neutral (#d97706)
- Modern cards, top header, left navigation sidebar, KPI badges
"""

import streamlit as st


def apply_custom_styles():
    """Inject modern financial analytics custom CSS styles into Streamlit app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Light Neutral Financial Background */
        .stApp {
            background-color: #f8fafc;
            color: #0f172a;
        }

        /* Top Header Styling */
        .top-header-container {
            background: #ffffff;
            border-bottom: 1px solid #e2e8f0;
            padding: 1.25rem 2rem;
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }

        .brand-title {
            font-size: 2rem;
            font-weight: 800;
            color: #1e1b4b;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .brand-subtitle {
            font-size: 1rem;
            font-weight: 600;
            color: #4f46e5;
            margin-top: 0.1rem;
        }

        .brand-sec-subtitle {
            font-size: 0.85rem;
            color: #64748b;
            margin-top: 0.1rem;
        }

        /* Status Badges */
        .badge-demo {
            background: #fef3c7;
            color: #92400e;
            border: 1px solid #fcd34d;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.75rem;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        .badge-live {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.75rem;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        .badge-positive {
            background: #dcfce7;
            color: #15803d;
            font-weight: 700;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
        }

        .badge-negative {
            background: #fee2e2;
            color: #b91c1c;
            font-weight: 700;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
        }

        .badge-neutral {
            background: #f1f5f9;
            color: #475569;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
        }

        /* Analytics Card Container */
        .analytics-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            transition: border-color 0.2s ease;
        }

        .analytics-card:hover {
            border-color: #cbd5e1;
        }

        /* Top 3 Portfolio Cards */
        .rank-card {
            background: #ffffff;
            border: 2px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }

        .rank-card-1 {
            border-color: #f59e0b;
            background: linear-gradient(180deg, #fffbeb 0%, #ffffff 100%);
        }

        .rank-card-2 {
            border-color: #94a3b8;
            background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
        }

        .rank-card-3 {
            border-color: #d97706;
            background: linear-gradient(180deg, #fff7ed 0%, #ffffff 100%);
        }

        /* Built-In Explanation Callout Box Pattern */
        .explanation-callout {
            background: #f0f9ff;
            border-left: 4px solid #0284c7;
            border-radius: 0 8px 8px 0;
            padding: 1rem 1.25rem;
            margin: 1rem 0;
            color: #0c4a6e;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .explanation-title {
            font-weight: 700;
            color: #0369a1;
            margin-bottom: 0.25rem;
            font-size: 1rem;
        }

        /* Pipeline Visual Nodes */
        .pipeline-container {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }

        .pipeline-node {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            text-align: center;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
            transition: transform 0.2s ease;
        }

        .pipeline-node:hover {
            border-color: #4f46e5;
            background: #eeef2a10;
            transform: translateY(-2px);
        }

        .pipeline-step-num {
            font-size: 0.75rem;
            font-weight: 700;
            color: #4f46e5;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }

        .pipeline-node-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }

        .pipeline-node-caption {
            font-size: 0.8rem;
            color: #64748b;
            line-height: 1.25;
        }

        /* Data Source Error Card */
        .data-error-card {
            background: #fef2f2;
            border: 1px solid #fca5a5;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            color: #991b1b;
        }

        /* Table & DataFrame Styling */
        .stDataFrame {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }

        @media (max-width: 768px) {
            .top-header-container {
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
