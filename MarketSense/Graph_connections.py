import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta

st.set_page_config(page_title="Dynamic Financial Infrastructure Risk", layout="wide")

# --- 1. DATA INGESTION: DYNAMIC CORRELATION ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_live_correlation():
    bank_tickers = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"]
    # Fetch 1 year of data for a stable correlation baseline
    data = yf.download(bank_tickers, period="1y", interval="1d")['Close']
    returns = data.pct_change().dropna()
    return returns.corr(), bank_tickers

corr_matrix, bank_names = get_live_correlation()

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Network Stress Controller")
target_bank = "HDFCBANK.NS"
hdfc_leverage = st.sidebar.slider("HDFC Leverage", 1.0, 5.0, 1.0)
hdfc_shadow_risk = st.sidebar.slider("HDFC Shadow Risk Gap", 0.0, 1.0, 0.05)

# --- 3. DYNAMIC SIMULATION LOGIC ---
def calculate_contagion(lev, shadow, matrix, tickers):
    signal = lev * shadow
    impacts = {}
    for bank in tickers:
        if bank == target_bank:
            impacts[bank] = signal
        else:
            # Use the LIVE correlation value as the spillover multiplier
            correlation = matrix.loc[target_bank, bank]
            impacts[bank] = (signal * correlation * 0.8) + 0.1
            
    avg_stress = np.mean(list(impacts.values()))
    ccp_margin = 0.02 + (avg_stress * 0.15)
    return impacts, ccp_margin, avg_stress

impacts, ccp_margin, avg_stress = calculate_contagion(hdfc_leverage, hdfc_shadow_risk, corr_matrix, bank_names)

# --- 4. VISUALIZATION: THE NETWORK GRAPH ---
def draw_network(impact_dict, margin, matrix, tickers):
    G = nx.Graph()
    G.add_node("CCP", pos=(0,0), type='hub')
    
    # Banks in a circle
    for i, bank in enumerate(tickers):
        theta = 2 * np.pi * i / len(tickers)
        G.add_node(bank, pos=(np.cos(theta), np.sin(theta)), risk=impact_dict[bank])
        # Edge thickness based on real-time correlation to HDFC
        weight = matrix.loc["HDFCBANK.NS", bank] if bank != "HDFCBANK.NS" else 1.0
        G.add_edge("CCP", bank, weight=weight)

    pos = nx.get_node_attributes(G, 'pos')
    
    # Create Traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), mode='lines')

    node_x, node_y, node_color, node_text = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if node == "CCP":
            node_color.append("gold")
            node_text.append(f"CCP<br>Margin: {margin:.2%}")
        else:
            risk = G.nodes[node]['risk']
            node_color.append('red' if risk > 0.3 else 'lightgreen')
            node_text.append(f"{node.split('.')[0]}<br>Stress: {risk:.2f}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
        marker=dict(showscale=False, size=35, color=node_color, line_width=2)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(showlegend=False, xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)))
    return fig

# --- 5. DASHBOARD LAYOUT ---
st.title("🏦 Live Network-Based Financial Stress Model")
st.write(f"Correlation data synced as of: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(draw_network(impacts, ccp_margin, corr_matrix, bank_names), use_container_width=True)

with col2:
    st.metric("CCP Margin Requirement", f"{ccp_margin:.2%}", delta=f"{ccp_margin-0.02:.2%}", delta_color="inverse")
    st.write("### Live Correlation Matrix (Input Data)")
    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))

# --- 6. ADDING THE "HOW IT WORKS" DYNAMIC INFO ---
st.divider()
with st.expander("🔍 Systemic Logic: How these features propagate risk", expanded=True):
    st.write(f"### The Chain Reaction of HDFC's {hdfc_leverage}x Leverage")
    
    # Dynamic Explanation Logic
    if hdfc_leverage > 3.0 or hdfc_shadow_risk > 0.4:
        st.warning(f"""
        **1. Micro-Level (HDFC):** By setting leverage to **{hdfc_leverage}x**, HDFC is maximizing its profit potential but thinning its capital buffer. 
        The Shadow Risk Gap of **{hdfc_shadow_risk}** indicates high hidden volatility.
        
        **2. Network Propagation:** Because HDFC is highly correlated with ICICI ({corr_matrix.loc["HDFCBANK.NS", "ICICIBANK.NS"]:.2f}) 
        and Axis ({corr_matrix.loc["HDFCBANK.NS", "AXISBANK.NS"]:.2f}), its instability is being "exported" to them.
        
        **3. Macro-Level (CCP):** The CCP detects a Total Systemic Stress of **{avg_stress:.2f}**. 
        To prevent a cascading failure, it has hiked the Margin Requirement to **{ccp_margin:.2%}.**
        
        **4. Impact on Others:** ICICI and Axis now face higher costs of capital (Margins) purely because of HDFC's aggressive features. 
        This is an **Incentive Misalignment** in the shared infrastructure.
        """)
    else:
        st.success("""
        **System Status: Stable.** Current HDFC features are within 'Resilient' thresholds. The CCP maintains a baseline margin, 
        and other banks are not suffering from contagion effects.
        """)

# Optional: Add a small technical diagram/flowchart explanation
st.caption("Flow: [HDFC Feature Change] -> [Network Volatility Spillover] -> [CCP Margin Intervention] -> [Other Bank Payoff Drop]")