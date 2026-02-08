# MarketSense-Lumina

A real-time risk management platform that utilizes **Multi-Agent Reinforcement Learning (MARL)** and **Network Contagion Theory** to predict and mitigate systemic financial risk.

## Overview

MarketSense-Lumina moves beyond static risk reporting. It simulates a living financial ecosystem where bank "agents" learn optimal leverage strategies through Reinforcement Learning while a Central Counterparty (CCP) dynamically adjusts margin requirements to prevent systemic collapse.

---

## Core Implementation Logic

The project features two primary engines that work in tandem to model systemic resilience:

### 1. Stability-Seeking MARL Engine

This engine simulates 10 major Indian banking tickers (HDFC, ICICI, SBIN, etc.) as autonomous RL agents.

* **Adaptive Leverage:** Agents adjust leverage based on "Network Heat" and market volatility.
* **Collective Penalty:** If the mean leverage across all agents exceeds safety thresholds, the CCP triggers a global margin hike, penalizing the rewards of all agents.
* **Predictive De-leveraging:** Agents learn to preemptively cut exposure when the India VIX rises or systemic stress accumulates.

### 2. Live Contagion Dashboard

A Streamlit-powered interactive digital twin of the banking network.

* **Dynamic Correlation:** Fetches live 1-year correlation matrices via `yfinance` to map interdependencies.
* **Shadow Risk Simulation:** Allows users to stress-test a "Target Bank" (e.g., HDFC) and observe real-time risk spillover to peer banks.
* **CCP Intervention Logic:** The system automatically calculates the required margin hike needed to offset current network stress based on real-time volatility.

---

## Features

* **RL-Driven Agent Modeling**: Autonomous agents that optimize for profit while avoiding systemic "instability penalties."
* **Real-Time Network Graph**: Interactive visualization of bank-to-CCP connections with edge weights based on live correlations.
* **Automated Margin Policy**: Simulation of CCP behavior that hikes margins incrementally during high-stress "Network Heat" events.
* **Predictive Dashboards**: Metric-driven insights into how one bank's leverage affects the entire infrastructure's cost of capital.

---

## Architecture

The system follows a four-phase implementation approach:

1. **Phase I: Market Oracle**: Real-time ingestion of NSE/BSE data and VIX signals.
2. **Phase II: MARL Engine**: Training agents using a dynamic reward function:


3. **Phase III: Contagion Mapping**: Applying live correlation matrices to a hub-and-spoke NetworkX model.
4. **Phase IV: Intervention UI**: Streamlit dashboard for real-time "What-If" scenario analysis.

---

## Installation

### Prerequisites

* Python 3.9+
* Dependencies: `yfinance`, `pandas`, `numpy`, `matplotlib`, `plotly`, `streamlit`, `networkx`

### Setup

1. Clone the repository:
```bash
git clone https://github.com/gauravmatolia/datathon26.git
cd datathon26/MarketSense

```


2. Install dependencies:
```bash
pip install -r requirements.txt

```



---

## Usage

### Running the MARL Simulation

To view the animated RL agent behavior and dynamic CCP margin spikes:

Run the first codeblock in the RLSimulation_Model.ipynb

### Launching the Live Risk Dashboard

To interact with the network stress-tester and correlation matrix:

```bash
streamlit run Graph_connections.py

```

---

## Systemic Logic: The Chain Reaction

1. **Micro-Level:** An agent (Bank) increases leverage to boost returns.
2. **Network Propagation:** High leverage combined with high market correlation "exports" volatility to the CCP.
3. **Macro-Intervention:** The CCP detects systemic stress and hikes margins.
4. **Feedback Loop:** Higher margins increase the cost of capital, forcing RL agents to de-leverage, eventually stabilizing the system.

---

**Would you like me to include a specific technical section detailing the Q-Learning Bellman equation used for the agent training?**