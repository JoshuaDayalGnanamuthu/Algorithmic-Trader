from Historicals import model, X_val, future_val
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

predictions = model.predict(X_val)
capital = 100000 
current_capital = capital
equity_curve = [capital]
trades_taken = 0
total_trades = [trades_taken]
wins = 0
losses = 0
avg_wins = []
avg_losses = []
binary_preds = (predictions > 0.45).astype(int)

for i in range(len(binary_preds)):
    if binary_preds[i] == 1:
        trading_capital = 0.05 * current_capital
        current_capital -= trading_capital
        trades_taken += 1
        trade_return = future_val[i] 
        trading_capital *= (1 + trade_return)
        current_capital += trading_capital
        
        if trade_return > 0:
            wins += 1
            avg_wins.append(trade_return)
        else:
            losses += 1
            avg_losses.append(trade_return)
    equity_curve.append(current_capital)
    total_trades.append(trades_taken)

# --- Summary Statistics ---
total_return_pct = ((current_capital - capital) / capital) * 100
win_rate = (wins / trades_taken * 100) if trades_taken > 0 else 0
peak = np.maximum.accumulate(equity_curve)
drawdown = (equity_curve - peak) / peak
max_drawdown = np.min(drawdown) * 100
avg_win_rate = mean(avg_wins)
avg_loss_rate = mean(avg_losses)
print(f"Max Drawdown:      {max_drawdown:.2f}%")

print("\n" + "="*30)
print("      BACKTEST RESULTS")
print("="*30)
print(f"Final Capital:     ${current_capital:,.2f}")
print(f"Total Return:      {total_return_pct:.2f}%")
print(f"Trades Executed:   {trades_taken}")
print(f"Trade Win Rate:    {win_rate:.2f}%")
print(f"Average Profit: {100 * avg_win_rate:.2f}%")
print(f"Average Loss: {100 * avg_loss_rate:.2f}%")
print(f"Average Value per Trade: {(win_rate/100 * avg_win_rate * 100) - ((1 - win_rate/100) * avg_loss_rate * 100)}%\n")
print("="*30)

plt.plot(total_trades, equity_curve)
plt.title("Equity Curve - Model Performance")
plt.show()