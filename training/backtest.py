from ModularNeuralNetwork import ModularNeuralNet
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from statistics import mean, stdev
import numpy as np
from config import PATHS
import xgboost as xgb

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[38;5;82m"
    RED    = "\033[38;5;196m"
    YELLOW = "\033[38;5;220m"
    CYAN   = "\033[38;5;51m"
    WHITE  = "\033[38;5;255m"
    GRAY   = "\033[38;5;240m"
    BG_DARK = "\033[48;5;234m"

def col(text, *codes):
    return "".join(codes) + str(text) + C.RESET

def signed_col(value, fmt=".2f"):
    color = C.GREEN if value >= 0 else C.RED
    sign  = "+" if value >= 0 else ""
    return col(f"{sign}{value:{fmt}}", color, C.BOLD)

def load_artifacts(model_path=PATHS["model"], x_path=PATHS["X_val"], future_path=PATHS["future"]):
    # model = xgb.XGBClassifier()
    # model.load_model(model_path)
    model        = ModularNeuralNet.load_model(model_path)
    X_validate     = np.load(x_path)
    future_returns = np.load(future_path)
    return model, X_validate, future_returns

def generate_signals(model, X, threshold=0.55):
    raw_preds    = model.predict(X, threshold = threshold)
    probabilities = model.predict_probability(X).flatten()
    signals      = (raw_preds.flatten() > threshold).astype(int)
    return signals, probabilities
    # probabilities = model.predict_proba(X)[:, 1]
    # signals = (probabilities > threshold).astype(int)
    # return signals, probabilities

def run_backtest(signals, future_returns, capital=100_000, position_size=0.05):
    current_capital = capital
    equity_curve    = [capital]
    trade_indices   = []
    wins, losses    = [], []

    for i, signal in enumerate(signals):
        if signal == 1:
            trade_capital  = position_size * current_capital
            trade_return   = future_returns[i]
            pnl            = trade_capital * trade_return
            current_capital += pnl
            trade_indices.append(i)

            if trade_return > 0:
                wins.append(trade_return)
            else:
                losses.append(trade_return)

        equity_curve.append(current_capital)

    return {
        "equity_curve":  np.array(equity_curve),
        "trade_indices": trade_indices,
        "wins":          wins,
        "losses":        losses,
        "initial":       capital,
        "final":         current_capital,
    }

def compute_metrics(result):
    eq     = result["equity_curve"]
    wins   = result["wins"]
    losses = result["losses"]
    n      = len(result["trade_indices"])

    total_return  = (result["final"] - result["initial"]) / result["initial"] * 100
    win_rate      = len(wins) / n * 100 if n > 0 else 0
    avg_win       = mean(wins)   * 100 if wins   else 0
    avg_loss      = mean(losses) * 100 if losses else 0
    expectancy    = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * abs(avg_loss))

    peak          = np.maximum.accumulate(eq)
    drawdowns     = (eq - peak) / peak * 100
    max_drawdown  = np.min(drawdowns)
    returns       = np.diff(eq) / eq[:-1]
    sharpe        = (np.mean(returns) / np.std(returns) * np.sqrt(252 * 6.5)
                     if np.std(returns) > 0 else 0)

    calmar        = (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0
    gross_profit  = sum(wins)
    gross_loss    = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    win_std  = stdev(wins)  * 100 if len(wins)   > 1 else 0
    loss_std = stdev(losses)* 100 if len(losses) > 1 else 0

    return {
        "total_return":  total_return,
        "win_rate":      win_rate,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "win_std":       win_std,
        "loss_std":      loss_std,
        "expectancy":    expectancy,
        "max_drawdown":  max_drawdown,
        "sharpe":        sharpe,
        "calmar":        calmar,
        "profit_factor": profit_factor,
        "n_trades":      n,
        "n_wins":        len(wins),
        "n_losses":      len(losses),
        "drawdowns":     drawdowns,
    }

def print_report(result, metrics, probabilities):
    W = 54
    bar = col("─" * W, C.GRAY)
    thick_bar = col("═" * W, C.CYAN)

    def row(label, value, width=W):
        pad = width - len(label) - len(_strip_ansi(str(value))) - 2
        return f"  {col(label, C.DIM)}{'·' * max(pad, 1)}{value}"

    def section(title):
        title_str = f"  {col(title, C.BOLD, C.WHITE)}  "
        side = (W - len(_strip_ansi(title_str))) // 2
        return f"{col('─' * side, C.GRAY)}{title_str}{col('─' * side, C.GRAY)}"

    m = metrics

    print()
    print(thick_bar)
    print(col(f"{'BACKTEST REPORT':^{W}}", C.BOLD, C.CYAN))
    print(thick_bar)

    print(section("PERFORMANCE"))
    print(row("Final Capital",   col(f"${result['final']:>12,.2f}", C.WHITE, C.BOLD)))
    print(row("Total Return",    signed_col(m['total_return'])))
    print(row("Max Drawdown",    col(f"{m['max_drawdown']:.2f}%",  C.RED)))
    print(row("Sharpe Ratio",    col(f"{m['sharpe']:.3f}",         C.YELLOW)))
    print(row("Calmar Ratio",    col(f"{m['calmar']:.3f}",         C.YELLOW)))
    print(row("Profit Factor",   col(f"{m['profit_factor']:.3f}",  C.YELLOW)))
    print(bar)

    print(section("TRADE STATISTICS"))
    print(row("Trades Executed",  col(f"{m['n_trades']}", C.WHITE)))
    print(row("Win / Loss",       col(f"{m['n_wins']} W  /  {m['n_losses']} L", C.WHITE)))
    print(row("Win Rate",         col(f"{m['win_rate']:.2f}%", C.GREEN if m['win_rate'] >= 50 else C.RED)))
    print(row("Avg Win",          col(f"+{m['avg_win']:.3f}%  (±{m['win_std']:.3f}%)", C.GREEN)))
    print(row("Avg Loss",         col(f"{m['avg_loss']:.3f}%  (±{m['loss_std']:.3f}%)", C.RED)))
    print(row("Expectancy / Trade", signed_col(m['expectancy'], ".3f") + col("%", C.RESET)))
    print(bar)

    active_probs = probabilities[np.array(result["trade_indices"])] if result["trade_indices"] else np.array([])
    if len(active_probs) > 0:
        print(section("MODEL CONFIDENCE"))
        print(row("Mean Confidence",   col(f"{active_probs.mean()*100:.1f}%", C.WHITE)))
        print(row("Min Confidence",    col(f"{active_probs.min()*100:.1f}%",  C.GRAY)))
        print(row("Max Confidence",    col(f"{active_probs.max()*100:.1f}%",  C.WHITE)))
        print(bar)

    print(thick_bar)
    print()

def _strip_ansi(text):
    """Remove ANSI codes for length calculation."""
    import re
    return re.sub(r'\033\[[0-9;]*m', '', text)

def plot_results(result, metrics, probabilities):
    eq      = result["equity_curve"]
    dd      = metrics["drawdowns"]
    trades  = result["trade_indices"]
    n       = len(eq)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 8), facecolor="#0d0d0d")
    fig.suptitle("Backtest Analysis", fontsize=15, fontweight="bold",
                 color="#e0e0e0", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3,
                           height_ratios=[3, 1.2, 1.2])

    ax_eq  = fig.add_subplot(gs[0, :]) 
    ax_dd  = fig.add_subplot(gs[1, :]) 
    ax_ret = fig.add_subplot(gs[2, 0])
    ax_conf= fig.add_subplot(gs[2, 1]) 

    ACCENT = "#00e5ff"
    RED    = "#ff3b5c"
    GREEN  = "#00e676"
    MUTED  = "#37474f"

    x_eq = np.arange(n)
    ax_eq.plot(x_eq, eq, color=ACCENT, linewidth=1.4, zorder=3)
    ax_eq.fill_between(x_eq, eq, eq.min(), alpha=0.08, color=ACCENT)
    ax_eq.set_facecolor("#0d0d0d")
    ax_eq.set_title("Equity Curve", color="#aaaaaa", fontsize=10, pad=6)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_eq.tick_params(colors="#555555", labelsize=8)
    for spine in ax_eq.spines.values():
        spine.set_edgecolor("#222222")

    if trades:
        trade_eq = [eq[t] for t in trades]
        ax_eq.scatter(trades, trade_eq, color=ACCENT, s=6, alpha=0.4, zorder=4)

    ax_dd.fill_between(x_eq, dd, 0, color=RED, alpha=0.5)
    ax_dd.plot(x_eq, dd, color=RED, linewidth=0.8)
    ax_dd.set_facecolor("#0d0d0d")
    ax_dd.set_title("Drawdown %", color="#aaaaaa", fontsize=10, pad=6)
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax_dd.tick_params(colors="#555555", labelsize=8)
    for spine in ax_dd.spines.values():
        spine.set_edgecolor("#222222")

    wins_arr  = np.array(result["wins"])   * 100
    losses_arr= np.array(result["losses"]) * 100
    if len(wins_arr):
        ax_ret.hist(wins_arr,   bins=20, color=GREEN, alpha=0.7, label="Wins")
    if len(losses_arr):
        ax_ret.hist(losses_arr, bins=20, color=RED,   alpha=0.7, label="Losses")
    ax_ret.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax_ret.set_facecolor("#0d0d0d")
    ax_ret.set_title("Trade Return Distribution", color="#aaaaaa", fontsize=10, pad=6)
    ax_ret.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax_ret.tick_params(colors="#555555", labelsize=8)
    ax_ret.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#333333", labelcolor="#aaaaaa")
    for spine in ax_ret.spines.values():
        spine.set_edgecolor("#222222")

    active_probs = (probabilities[np.array(trades)] * 100
                    if trades else np.array([]))
    if len(active_probs):
        ax_conf.hist(active_probs, bins=20, color=ACCENT, alpha=0.7)
        ax_conf.axvline(active_probs.mean(), color="#ffffff", linewidth=1,
                        linestyle="--", label=f"Mean {active_probs.mean():.1f}%")
        ax_conf.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#333333",
                       labelcolor="#aaaaaa")
    ax_conf.set_facecolor("#0d0d0d")
    ax_conf.set_title("Signal Confidence Distribution", color="#aaaaaa", fontsize=10, pad=6)
    ax_conf.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_conf.tick_params(colors="#555555", labelsize=8)
    for spine in ax_conf.spines.values():
        spine.set_edgecolor("#222222")

    # plt.savefig("backtest_chart.png", dpi=150, bbox_inches="tight",
    #             facecolor="#0d0d0d")
    plt.show()

def main():
    model, X_val, future_returns = load_artifacts()
    signals, probabilities       = generate_signals(model, X_val, threshold=0.55)
    result                       = run_backtest(signals, future_returns)
    metrics                      = compute_metrics(result)

    print_report(result, metrics, probabilities)
    plot_results(result, metrics, probabilities)

if __name__ == "__main__":
    main()