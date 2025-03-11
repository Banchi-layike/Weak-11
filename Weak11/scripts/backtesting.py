import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


class Backtester:
    def __init__(self, symbol, start="2020-01-01", end="2024-01-01", initial_cash=10000, transaction_cost=0.001):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.data = self.fetch_data()

    def fetch_data(self):
        """Fetch historical stock data with error handling"""
        df = yf.download(self.symbol, start=self.start, end=self.end)
        
        if "Adj Close" not in df.columns:
            print("⚠️ Warning: 'Adj Close' column not found. Using 'Close' instead.")
            df["Adj Close"] = df["Close"]

        df["Returns"] = df["Adj Close"].pct_change()
        df.dropna(inplace=True)
        return df

    def moving_average_crossover(self, short_window=50, long_window=200):
        df = self.data.copy()
        df["Short_MA"] = df["Adj Close"].rolling(window=short_window).mean()
        df["Long_MA"] = df["Adj Close"].rolling(window=long_window).mean()
        df["Signal"] = np.where(df["Short_MA"] > df["Long_MA"], 1, 0)
        df["Position"] = df["Signal"].diff()
        return self.simulate_trading(df, "Moving Average Crossover")

    def rsi_strategy(self, period=14, overbought=70, oversold=30):
        df = self.data.copy()
        delta = df["Adj Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["Signal"] = np.where(df["RSI"] < oversold, 1, np.where(df["RSI"] > overbought, -1, 0))
        df["Position"] = df["Signal"].diff()
        return self.simulate_trading(df, "RSI Strategy")

    def bollinger_bands(self, window=20, num_std=2):
        df = self.data.copy()
        df["Rolling_Mean"] = df["Adj Close"].rolling(window=window).mean()
        df["Rolling_Std"] = df["Adj Close"].rolling(window=window).std()
        df["Upper_Band"] = df["Rolling_Mean"] + (df["Rolling_Std"] * num_std)
        df["Lower_Band"] = df["Rolling_Mean"] - (df["Rolling_Std"] * num_std)
        df["Signal"] = np.where(df["Adj Close"] < df["Lower_Band"], 1, np.where(df["Adj Close"] > df["Upper_Band"], -1, 0))
        df["Position"] = df["Signal"].diff()
        return self.simulate_trading(df, "Bollinger Bands Strategy")

    def macd_strategy(self):
        df = self.data.copy()
        df["MACD"] = df["Adj Close"].ewm(span=12, adjust=False).mean() - df["Adj Close"].ewm(span=26, adjust=False).mean()
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["Signal"] = np.where(df["MACD"] > df["Signal_Line"], 1, np.where(df["MACD"] < df["Signal_Line"], -1, 0))
        df["Position"] = df["Signal"].diff()
        return self.simulate_trading(df, "MACD Crossover Strategy")

    def breakout_strategy(self, window=20):
        df = self.data.copy()
        df["High_Max"] = df["Adj Close"].rolling(window=window).max()
        df["Low_Min"] = df["Adj Close"].rolling(window=window).min()
        df["Signal"] = np.where(df["Adj Close"] > df["High_Max"].shift(1), 1, np.where(df["Adj Close"] < df["Low_Min"].shift(1), -1, 0))
        df["Position"] = df["Signal"].diff()
        return self.simulate_trading(df, "Breakout Strategy")

    def mean_reversion_strategy(self, window=20, threshold=1.5):
        df = self.data.copy()
        df["Rolling_Mean"] = df["Adj Close"].rolling(window=window).mean()
        df["Rolling_Std"] = df["Adj Close"].rolling(window=window).std()
        df["Upper_Band"] = df["Rolling_Mean"] + threshold * df["Rolling_Std"]
        df["Lower_Band"] = df["Rolling_Mean"] - threshold * df["Rolling_Std"]
        df["Signal"] = np.where(df["Adj Close"] < df["Lower_Band"], 1, np.where(df["Adj Close"] > df["Upper_Band"], -1, 0))
        df["Position"] = df["Signal"].diff()
        return self.simulate_trading(df, "Mean Reversion Strategy")

    def simulate_trading(self, df, strategy_name):
        cash = self.initial_cash
        position = 0
        portfolio_value = []
        trades = 0
        win_trades = 0
        drawdowns = []
        entry_price = None

        for i in range(len(df)):
            price = df["Adj Close"].iloc[i]
            signal = df["Signal"].iloc[i]

            if signal == 1 and cash > price:
                num_shares = cash // price
                cash -= num_shares * price * (1 + self.transaction_cost)
                position += num_shares
                entry_price = price
                trades += 1

            elif signal == -1 and position > 0:
                proceeds = position * price * (1 - self.transaction_cost)
                if price > entry_price:
                    win_trades += 1
                cash += proceeds
                position = 0

            portfolio_value.append(cash + position * price)
            max_value = max(portfolio_value) if portfolio_value else self.initial_cash
            drawdowns.append((portfolio_value[-1] - max_value) / max_value)

        df["Portfolio Value"] = portfolio_value
        df["Drawdown"] = drawdowns

        total_return = df["Portfolio Value"].iloc[-1] / self.initial_cash - 1
        sharpe_ratio = df["Returns"].mean() / df["Returns"].std() * np.sqrt(252)
        max_drawdown = df["Drawdown"].min()
        win_rate = win_trades / trades if trades > 0 else 0

        return {
            "Strategy": strategy_name,
            "Total Return": f"{total_return:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Win Rate": f"{win_rate:.2%}",
            "Total Trades": trades
        }
    def simulate_trading(self, df, strategy_name):
        """Simulate trading performance and visualize results"""
        cash = self.initial_cash
        position = 0
        portfolio_value = []
        trades = 0
        win_trades = 0
        drawdowns = []
        entry_price = None

        buy_signals = []
        sell_signals = []

        for i in range(len(df)):
            price = df["Adj Close"].iloc[i]
            signal = df["Signal"].iloc[i]

            if signal == 1 and cash > price:
                num_shares = cash // price
                cash -= num_shares * price * (1 + self.transaction_cost)
                position += num_shares
                entry_price = price
                trades += 1
                buy_signals.append(price)
                sell_signals.append(np.nan)
            elif signal == -1 and position > 0:
                proceeds = position * price * (1 - self.transaction_cost)
                if price > entry_price:
                    win_trades += 1
                cash += proceeds
                position = 0
                buy_signals.append(np.nan)
                sell_signals.append(price)
            else:
                buy_signals.append(np.nan)
                sell_signals.append(np.nan)

            portfolio_value.append(cash + position * price)
            max_value = max(portfolio_value) if portfolio_value else self.initial_cash
            drawdowns.append((portfolio_value[-1] - max_value) / max_value)

        df["Portfolio Value"] = portfolio_value
        df["Drawdown"] = drawdowns

        total_return = df["Portfolio Value"].iloc[-1] / self.initial_cash - 1
        sharpe_ratio = df["Returns"].mean() / df["Returns"].std() * np.sqrt(252)
        max_drawdown = df["Drawdown"].min()
        win_rate = win_trades / trades if trades > 0 else 0

        # 🎯 PLOT 1: Buy/Sell Signals on Price Chart
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["Adj Close"], label="Stock Price", color="blue", alpha=0.7)
        plt.scatter(df.index, buy_signals, label="Buy Signal", marker="^", color="green", alpha=1)
        plt.scatter(df.index, sell_signals, label="Sell Signal", marker="v", color="red", alpha=1)
        plt.title(f"{strategy_name} - Buy/Sell Signals")
        plt.xlabel("Date")
        plt.ylabel("Stock Price ($)")
        plt.legend()
        plt.show()

        # 🎯 PLOT 2: Portfolio Value Over Time
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["Portfolio Value"], label="Portfolio Value", color="purple", linewidth=2)
        plt.title(f"Portfolio Growth - {strategy_name}")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.show()

        return {
            "Strategy": strategy_name,
            "Total Return": f"{total_return:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Win Rate": f"{win_rate:.2%}",
            "Total Trades": trades
        }
