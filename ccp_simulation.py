import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import random
from itertools import combinations

plt.switch_backend('Agg')
def safe_show(title):
    plt.savefig(f"{title}.png", bbox_inches='tight')
    print(f"Saved plot: {title}.png")
    plt.close()
# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CONFIG = {
    "num_traders": 6,
    "days": 40,

    "base_margin_rate": 0.10,
    "default_fund_rate": 0.05,
    "ccp_initial_capital": 1_000_000,

    "volatility": 0.02,
    "crash_probability": 0.02,
    "crash_size": -0.20,

    "interbank_lending_ratio": 0.25,   # % liquidity banks try to lend
    "repayment_probability": 0.97,     # probability loans are repaid daily

    "fear_sensitivity": 5.0,   # how strongly banks react to risk

    "lending_rate": 0.05,
    "trading_return_rate": 0.03,
    "risk_penalty": 0.20,
    "equilibrium_tolerance": 1e-3,

    "ccp_fee_rate": 0.02,
    "systemic_risk_penalty": 5.0
}

BANK_TICKERS = [
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "INDUSINDBK.NS"
]

def run_monte_carlo(n_runs=2):
    risks = []
    for i in range(n_runs):
        metrics, _ = run_simulation(CONFIG)
        risk = compute_systemic_risk(metrics)
        risks.append(risk)
    return np.mean(risks)

def compute_credit_score(ticker):
    try:
        data = yf.download(ticker, start="2020-01-01", progress=False)

        # Handle different yfinance column formats
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Use Adj Close if available, otherwise Close
        if "Adj Close" in data.columns:
            prices = data["Adj Close"]
        elif "Close" in data.columns:
            prices = data["Close"]
        else:
            return random.uniform(0.4, 0.6)  # fallback score

        returns = prices.pct_change().dropna()

        if len(returns) < 50:
            return random.uniform(0.4, 0.6)

        # Risk metrics
        mean_return = returns.mean()
        volatility = returns.std()

        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        var95 = np.percentile(returns, 5)

        raw_score = (
            2 * mean_return
            - 1.5 * volatility
            - 1.0 * abs(max_drawdown)
            - 1.0 * abs(var95)
        )

        credit_score = 1 / (1 + np.exp(-raw_score))
        return float(credit_score)

    except Exception as e:
        # If API fails, return neutral credit score
        print(f"Warning: Could not fetch {ticker}, using fallback score.")
        return random.uniform(0.4, 0.6)

class Trader:
    def __init__(self, trader_id, ticker):
        self.id = trader_id
        self.ticker = ticker
        self.name = f"{ticker}_Bank{trader_id}"
        
        self.capital = random.uniform(50000, 200000)
        self.risk_appetite = random.uniform(0.5, 2.0)
        self.credit_score = compute_credit_score(ticker)

        self.position = 0
        self.margin = 0
        self.defaulted = False

        # NEW: Interbank network variables
        self.liquidity = self.capital * CONFIG.get("initial_liquidity_ratio", 0.3)   # portion of capital kept as cash
        self.loans_given = {}   # money lent to other banks
        self.loans_taken = {}   # money borrowed

    def take_position(self):
        if self.defaulted:
            return

        self.position = (
            self.capital
            * self.credit_score
            * self.risk_appetite
            * random.uniform(-1, 1)
        )

    def update_pnl(self, market_return):
        if self.defaulted:
            return 0
        
        pnl = self.position * market_return
        self.capital += pnl
        return pnl

class ClearingHouse:
    def __init__(self, traders, margin_rate, default_fund_rate):
        self.traders = traders
        self.margin_rate = margin_rate
        self.default_fund_rate = default_fund_rate
        
        self.ccp_capital = CONFIG["ccp_initial_capital"]
        self.default_fund = 0

        for t in traders:
            contribution = t.capital * default_fund_rate
            self.default_fund += contribution

    def collect_margin(self):
        for t in self.traders:
            if t.defaulted:
                continue
            
            margin_rate_i = self.margin_rate + (1 - t.credit_score)
            t.margin = abs(t.position) * margin_rate_i

    def handle_default(self, trader, loss):
        remaining_loss = loss

        used_margin = min(trader.margin, remaining_loss)
        remaining_loss -= used_margin

        trader_df = trader.capital * self.default_fund_rate
        used_df = min(trader_df, remaining_loss)
        remaining_loss -= used_df

        used_ccp_df = min(self.default_fund, remaining_loss)
        self.default_fund -= used_ccp_df
        remaining_loss -= used_ccp_df

        if remaining_loss > 0:
            self.ccp_capital -= remaining_loss

        trader.defaulted = True

class Metrics:
    def __init__(self):
        self.daily_defaults = []
        self.daily_margin_calls = []
        self.ccp_capital_history = []
        self.market_returns = []
        self.system_liquidity = []
        self.system_defaults_ratio = []
        self.congestion_index = []

    def record_day(self, defaults, margin_calls, ccp_capital, market_return, traders):
        self.daily_defaults.append(defaults)
        self.daily_margin_calls.append(margin_calls)
        self.ccp_capital_history.append(ccp_capital)
        self.market_returns.append(market_return)

        # NEW SYSTEM METRICS
        total_liquidity = sum(t.liquidity for t in traders if not t.defaulted)
        self.system_liquidity.append(total_liquidity)

        default_ratio = sum(t.defaulted for t in traders) / len(traders)
        self.system_defaults_ratio.append(default_ratio)

        negative_liquidity = sum(1 for t in traders if t.liquidity < 0)
        self.congestion_index.append(negative_liquidity)

def market_return():
    if random.random() < CONFIG["crash_probability"]:
        return CONFIG["crash_size"]
    return np.random.normal(0, CONFIG["volatility"])

def create_interbank_network(traders, perceived_risk):
    n = len(traders)
    exposure_matrix = np.zeros((n, n))
    
    # Lending decreases when fear increases
    lending_multiplier = max(0, 1 - perceived_risk)

    for i, lender in enumerate(traders):
        for j, borrower in enumerate(traders):
            if i == j or lender.defaulted or borrower.defaulted:
                continue
            
            # Strategic lending decision
            if random.random() < 0.3 * lending_multiplier:
                
                loan_amount = (
                    lender.liquidity 
                    * CONFIG["interbank_lending_ratio"] 
                    * lending_multiplier
                    * random.random()
                )
                
                exposure_matrix[i][j] = loan_amount
                
                lender.loans_given[j] = loan_amount
                borrower.loans_taken[i] = loan_amount
                
                lender.liquidity -= loan_amount
                borrower.liquidity += loan_amount
    
    return exposure_matrix

def settle_interbank_loans(traders, exposure_matrix):
    n = len(traders)
    defaults = []

    for i in range(n):
        for j in range(n):
            loan = exposure_matrix[i][j]
            if loan == 0:
                continue
            
            # Borrower tries to repay lender
            if random.random() < CONFIG["repayment_probability"]:
                traders[j].liquidity -= loan
                traders[i].liquidity += loan
            else:
                # Loan default → lender loses money
                traders[i].liquidity -= loan
                defaults.append(j)

    return defaults

def check_liquidity_defaults(traders):
    new_defaults = 0
    for t in traders:
        if not t.defaulted and t.liquidity < 0:
            t.defaulted = True
            new_defaults += 1
    return new_defaults

def compute_perceived_risk(metrics):
    if len(metrics.market_returns) == 0:
        return 0.1
    
    recent_volatility = np.std(metrics.market_returns[-10:])
    recent_defaults = sum(metrics.daily_defaults[-10:]) if len(metrics.daily_defaults) > 0 else 0
    
    perceived_risk = recent_volatility * CONFIG["fear_sensitivity"] + (recent_defaults / 10)
    return perceived_risk

def compute_systemic_risk(metrics):
    avg_defaults = np.mean(metrics.system_defaults_ratio)
    avg_congestion = np.mean(metrics.congestion_index)
    avg_volatility = np.std(metrics.market_returns)

    risk_index = (
        0.5 * avg_defaults +
        0.3 * avg_congestion/len(metrics.congestion_index) +
        0.2 * avg_volatility
    )

    return risk_index

def compute_ccp_payoff(metrics, traders):
    total_trading = sum(abs(t.position) for t in traders)
    total_margin = sum(t.margin for t in traders)
    defaults = sum(metrics.daily_defaults)

    systemic_risk = compute_systemic_risk(metrics)

    trading_income = CONFIG["ccp_fee_rate"] * total_trading
    margin_income = 0.01 * total_margin
    default_losses = defaults * 50000
    systemic_penalty = CONFIG["systemic_risk_penalty"] * systemic_risk * 1e6

    payoff = trading_income + margin_income - default_losses - systemic_penalty
    return payoff

def find_ccp_equilibrium_margin():
    margin_candidates = np.linspace(0.02, 0.20, 10)
    payoffs = []

    print("\nSearching CCP equilibrium margin...")

    for m in margin_candidates:
        CONFIG["base_margin_rate"] = m
        metrics, traders = run_simulation(CONFIG)
        payoff = compute_ccp_payoff(metrics, traders)
        payoffs.append(payoff)

        print(f"Margin {m:.2f} → CCP payoff {payoff:.2f}")

    best_margin = margin_candidates[np.argmax(payoffs)]

    print("\nCCP EQUILIBRIUM FOUND")
    print("Optimal Margin Rate:", round(best_margin,3))

    plt.figure()
    plt.plot(margin_candidates, payoffs, marker='o')
    plt.title("CCP Payoff vs Margin (Equilibrium Search)")
    plt.xlabel("Margin Rate")
    plt.ylabel("CCP Payoff")
    safe_show("ccp_equilibrium")

    return best_margin

def simulate_day(traders, ccp, metrics):
    perceived_risk = compute_perceived_risk(metrics)
    defaults_today = 0
    margin_calls_today = 0

    # Traders take positions
    for t in traders:
        t.take_position()

    # Create interbank network ONCE per day
    exposure_matrix = create_interbank_network(traders, perceived_risk)

    # Settle interbank loans
    loan_defaults = settle_interbank_loans(traders, exposure_matrix)

    # Check liquidity failures
    defaults_from_network = check_liquidity_defaults(traders)


    ccp.collect_margin()

    r = market_return()

    for t in traders:
        if t.defaulted:
            continue

        pnl = t.update_pnl(r)

        if pnl < 0:
            margin_calls_today += 1

            if t.capital < abs(pnl):
                ccp.handle_default(t, abs(pnl))
                defaults_today += 1

    metrics.record_day(defaults_today, margin_calls_today, ccp.ccp_capital, r, traders)

def compute_pairwise_risk_matrix(traders):
    n = len(traders)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            bank_i = traders[i]
            bank_j = traders[j]

            # Credit risk difference
            credit_risk = abs(bank_i.credit_score - bank_j.credit_score)

            # Liquidity vulnerability
            liquidity_risk = max(0, (1 - bank_i.liquidity / (bank_i.capital + 1)))

            # Trading exposure proxy (margin usage)
            trading_risk = bank_i.margin / (bank_i.capital + 1)

            pair_risk = (
                0.4 * credit_risk +
                0.3 * liquidity_risk +
                0.3 * trading_risk
            )

            matrix[i][j] = pair_risk

    return matrix

def relationship_decision_matrix(traders, risk_matrix):
    decisions = []

    for i, bank_i in enumerate(traders):
        row = []
        for j, bank_j in enumerate(traders):
            if i == j:
                row.append("—")
                continue

            risk = risk_matrix[i][j]

            if risk < 0.25:
                decision = "KEEP"
            elif risk < 0.35:
                decision = "REDUCE"
            else:
                decision = "CUT"

            row.append(decision)

        decisions.append(row)

    bank_names = [t.ticker for t in traders]
    df = pd.DataFrame(decisions, index=bank_names, columns=bank_names)

    print("\nBANK RELATIONSHIP DECISION MATRIX\n")
    print(df)

    df.to_csv("relationship_decisions.csv")
    print("\nSaved table: relationship_decisions.csv")

def compute_payoff_matrix(traders, risk_matrix):
    n = len(traders)
    payoff_matrix = np.zeros((n, n))

    for i, bank_i in enumerate(traders):
        for j, bank_j in enumerate(traders):
            if i == j:
                continue

            lending_profit = CONFIG["lending_rate"] * bank_i.liquidity
            trading_profit = CONFIG["trading_return_rate"] * bank_i.margin
            risk_cost = CONFIG["risk_penalty"] * risk_matrix[i][j] * bank_i.capital

            payoff = lending_profit + trading_profit - risk_cost
            payoff_matrix[i][j] = payoff

    return payoff_matrix

def print_payoff_matrix(traders, payoff_matrix):
    print("\nBANK PAYOFF MATRIX\n")

    bank_names = [t.ticker for t in traders]

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    df = pd.DataFrame(payoff_matrix, index=bank_names, columns=bank_names)
    print(df.round(2))

    df.to_csv("payoff_matrix.csv")
    print("\nSaved table: payoff_matrix.csv")

def best_response_update(traders, risk_matrix):
    decisions_changed = False

    for i, bank_i in enumerate(traders):
        for j, bank_j in enumerate(traders):
            if i == j:
                continue

            risk = risk_matrix[i][j]

            # current payoff
            current_payoff = (
                CONFIG["lending_rate"] * bank_i.liquidity
                + CONFIG["trading_return_rate"] * bank_i.margin
                - CONFIG["risk_penalty"] * risk * bank_i.capital
            )

            # payoff if relation CUT (no profit, no risk)
            cut_payoff = 0

            if cut_payoff > current_payoff:
                # Bank prefers cutting relation
                decisions_changed = True

    return decisions_changed

def check_equilibrium(traders, risk_matrix, max_iter=10):
    print("\nChecking Clearing House Network Equilibrium...")

    for step in range(max_iter):
        changed = best_response_update(traders, risk_matrix)

        if not changed:
            print(f"Equilibrium reached after {step+1} iterations.")
            return True

    print("Equilibrium not reached (network still adjusting).")
    return False

def compute_bank_level_risk(traders, trader_matrix):
    tickers = list(set([t.ticker for t in traders]))
    bank_matrix = pd.DataFrame(0.0, index=tickers, columns=tickers)

    # Map traders to tickers
    ticker_to_indices = {}
    for i, t in enumerate(traders):
        ticker_to_indices.setdefault(t.ticker, []).append(i)

    # Average risk between banks
    for bank_i in tickers:
        for bank_j in tickers:
            indices_i = ticker_to_indices[bank_i]
            indices_j = ticker_to_indices[bank_j]

            risks = []
            for i in indices_i:
                for j in indices_j:
                    if i != j:
                        risks.append(trader_matrix[i][j])

            if len(risks) > 0:
                bank_matrix.loc[bank_i, bank_j] = np.mean(risks)

    return bank_matrix

def save_risk_matrix(matrix, labels, filename, title):
    print(f"\n{title}\n")

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # Check if matrix is a DataFrame or numpy array
    if isinstance(matrix, pd.DataFrame):
        df = matrix
    else:
        df = pd.DataFrame(matrix, index=labels, columns=labels)
    
    print(df.round(3))

    df.to_csv(filename)
    print(f"\nSaved table: {filename}")



def run_simulation(config=CONFIG):
    # Create exactly one trader per bank
    traders = []
    for i, ticker in enumerate(BANK_TICKERS):
        traders.append(Trader(i, ticker))

    ccp = ClearingHouse(
        traders,
        config["base_margin_rate"],
        config["default_fund_rate"]
    )

    metrics = Metrics()

    for day in range(config["days"]):
        simulate_day(traders, ccp, metrics)

        if ccp.ccp_capital < 0:
            print("CCP FAILED")
            break

    return metrics, traders

def margin_policy_experiment():
    margin_levels = [0.05, 0.10, 0.15, 0.20, 0.30]
    results = []
    
    print("\n--- Margin Policy Experiment Results ---")

    for m in margin_levels:
        # print(f"Testing margin = {m}") # Reduced verbosity
        CONFIG["base_margin_rate"] = m
        avg_risk = run_monte_carlo()
        results.append(avg_risk)
        print(f"Margin Rate: {m:.0%} -> Systemic Risk Index: {avg_risk:.4f}")

    plt.figure()
    plt.plot(margin_levels, results, marker='o')
    plt.title("Systemic Risk vs Margin Requirement")
    plt.xlabel("Margin Rate")
    plt.ylabel("Systemic Risk Index")

    # ⭐ Add values on points
    for x, y in zip(margin_levels, results):
        plt.text(x, y, f"{y:.3f}", fontsize=9)

    safe_show("margin_experiment")

def liquidity_policy_experiment():
    liquidity_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    for l in liquidity_levels:
        print(f"Testing liquidity buffer = {l}")
        for t in range(CONFIG["num_traders"]):
            pass
        
        CONFIG["initial_liquidity_ratio"] = l
        avg_risk = run_monte_carlo()
        results.append(avg_risk)

    plt.plot(liquidity_levels, results, marker='o')
    plt.title("Systemic Risk vs Liquidity Buffer")
    plt.xlabel("Liquidity Ratio")
    plt.ylabel("Systemic Risk Index")
    safe_show("liquidity_experiment")

def panic_experiment():
    fear_levels = [1, 3, 5, 7, 10]
    results = []

    for f in fear_levels:
        print(f"Testing fear sensitivity = {f}")
        CONFIG["fear_sensitivity"] = f
        avg_risk = run_monte_carlo()
        results.append(avg_risk)

    plt.plot(fear_levels, results, marker='o')
    plt.title("Systemic Risk vs Panic Sensitivity")
    plt.xlabel("Fear Sensitivity")
    plt.ylabel("Systemic Risk Index")
    safe_show("panic_experiment")

def plot_metrics(metrics):
    plt.figure(figsize=(14,10))

    plt.subplot(3,2,1)
    plt.plot(metrics.daily_defaults)
    plt.title("Daily Defaults")

    plt.subplot(3,2,2)
    plt.plot(metrics.daily_margin_calls)
    plt.title("Margin Calls")

    plt.subplot(3,2,3)
    plt.plot(metrics.ccp_capital_history)
    plt.title("CCP Capital")

    plt.subplot(3,2,4)
    plt.plot(metrics.system_liquidity)
    plt.title("System Liquidity")

    plt.subplot(3,2,5)
    plt.plot(metrics.system_defaults_ratio)
    plt.title("Default Ratio")

    plt.subplot(3,2,6)
    plt.plot(metrics.congestion_index)
    plt.title("Congestion Index")

    plt.tight_layout()
    safe_show("base_simulation")

if __name__ == "__main__":
    print("Running base simulation...")
    metrics, traders = run_simulation(CONFIG)
    plot_metrics(metrics)

    # ⭐ Pairwise TRADER COMPONENT risk table
    risk_matrix = compute_pairwise_risk_matrix(traders)
    trader_names = [t.name for t in traders]
    save_risk_matrix(risk_matrix, trader_names, "pairwise_risk_matrix.csv", "TRADER PAIRWISE RISK MATRIX")

    # ⭐ Aggregated BANK LEVEL risk table
    bank_risk_df = compute_bank_level_risk(traders, risk_matrix)
    save_risk_matrix(bank_risk_df, None, "bank_risk_matrix.csv", "BANK AGGREGATED RISK MATRIX")
    relationship_decision_matrix(traders, risk_matrix)

    # ⭐ PAYOFF MATRIX
    payoff_matrix = compute_payoff_matrix(traders, risk_matrix)
    print_payoff_matrix(traders, payoff_matrix)

    # ⭐ EQUILIBRIUM CHECK
    check_equilibrium(traders, risk_matrix)

    # ⭐ CCP equilibrium search
    find_ccp_equilibrium_margin()

    print("\nRunning experiments...")
    margin_policy_experiment()
    liquidity_policy_experiment()
    panic_experiment()

    risk_index = compute_systemic_risk(metrics)
    print("\nSystemic Risk Index:", round(risk_index,3))