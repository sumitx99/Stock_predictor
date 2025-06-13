import tkinter as tk
from tkinter import ttk, messagebox
from ttkbootstrap import Style, Notebook
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import datetime, timedelta
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class StockPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Market Predictor")
        self.style = Style(theme='cosmo')

        # Initialize data storage FIRST
        self.model = None
        self.data = None
        self.X = None
        self.news_api_key = 'f298b0e94334471ca29ce04f019bcd60'  # Replace with your NewsAPI key
        self.portfolio = {'cash': 100000, 'holdings': {}}  # Initialize portfolio first

        # Create UI components
        self.notebook = Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        # Create tabs
        self.prediction_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        self.portfolio_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.prediction_tab, text='Prediction')
        self.notebook.add(self.visualization_tab, text='Visualization')
        self.notebook.add(self.portfolio_tab, text='Portfolio')

        # Build UI elements
        self.create_prediction_tab()
        self.create_visualization_tab()
        self.create_portfolio_tab()  # Now portfolio exists when this is called

    def create_prediction_tab(self):
        # Input frame
        input_frame = ttk.LabelFrame(self.prediction_tab, text="Input")
        input_frame.pack(padx=10, pady=10, fill='x')

        ttk.Label(input_frame, text="Stock Symbol:").grid(row=0, column=0, padx=5, pady=5)
        self.symbol_entry = ttk.Entry(input_frame, width=10)
        self.symbol_entry.grid(row=0, column=1, padx=5, pady=5)

        self.train_btn = ttk.Button(input_frame, text="Train Model", command=self.train_model)
        self.train_btn.grid(row=0, column=2, padx=5, pady=5)

        self.predict_btn = ttk.Button(input_frame, text="Predict", command=self.predict_tomorrow, state='disabled')
        self.predict_btn.grid(row=0, column=3, padx=5, pady=5)

        # Results frame
        results_frame = ttk.LabelFrame(self.prediction_tab, text="Results")
        results_frame.pack(padx=10, pady=10, fill='both', expand=True)

        self.result_tree = ttk.Treeview(results_frame, columns=('metric', 'value'), show='headings')
        self.result_tree.heading('metric', text='Metric')
        self.result_tree.heading('value', text='Value')
        self.result_tree.pack(fill='both', expand=True)

        # Status bar
        self.status_var = tk.StringVar()
        ttk.Label(self.prediction_tab, textvariable=self.status_var).pack(pady=5)

    def create_visualization_tab(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.visualization_tab)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def create_portfolio_tab(self):
        portfolio_frame = ttk.LabelFrame(self.portfolio_tab, text="Portfolio Management")
        portfolio_frame.pack(padx=10, pady=10, fill='x')

        ttk.Label(portfolio_frame, text="Symbol:").grid(row=0, column=0, padx=5, pady=5)
        self.portfolio_symbol = ttk.Entry(portfolio_frame, width=10)
        self.portfolio_symbol.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(portfolio_frame, text="Shares:").grid(row=0, column=2, padx=5, pady=5)
        self.portfolio_shares = ttk.Entry(portfolio_frame, width=10)
        self.portfolio_shares.grid(row=0, column=3, padx=5, pady=5)

        ttk.Button(portfolio_frame, text="Buy", command=self.buy_stock).grid(row=0, column=4, padx=5, pady=5)
        ttk.Button(portfolio_frame, text="Sell", command=self.sell_stock).grid(row=0, column=5, padx=5, pady=5)

        self.portfolio_tree = ttk.Treeview(self.portfolio_tab, columns=('asset', 'value'), show='headings')
        self.portfolio_tree.heading('asset', text='Asset')
        self.portfolio_tree.heading('value', text='Value')
        self.portfolio_tree.pack(fill='both', expand=True, padx=10, pady=10)

        self.update_portfolio_display()  # Now portfolio is initialized

    def fetch_data(self, symbol):
        try:
            self.status(f"Fetching data for {symbol}...")
            stock = yf.Ticker(symbol)
            hist = stock.history(period="2y")
            self.status("Data fetched successfully", color='green')
            return hist
        except Exception as e:
            self.status(f"Error fetching data: {str(e)}", color='red')
            return None

    def preprocess_data(self, data, symbol):
        # Technical indicators
        data['Return'] = data['Close'].pct_change()
        data['MA_5'] = data['Close'].rolling(5).mean()
        data['MA_20'] = data['Close'].rolling(20).mean()
        data['Volatility'] = data['Return'].rolling(5).std()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['MACD'], data['MACD_signal'] = self.calculate_macd(data['Close'])
        data['BB_upper'], data['BB_lower'] = self.calculate_bollinger_bands(data['Close'])

        # Sentiment analysis
        data['Sentiment'] = self.get_news_sentiment(symbol)

        data.dropna(inplace=True)
        X = data[['MA_5', 'MA_20', 'Volatility', 'RSI', 'MACD', 'BB_upper', 'Sentiment']]
        y = data['Close'].shift(-1)[:-1]
        X = X[:-1]
        return X, y

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, series, slow=26, fast=12, signal=9):
        macd = series.ewm(span=fast).mean() - series.ewm(span=slow).mean()
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    def calculate_bollinger_bands(self, series, window=20):
        ma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        return upper, lower

    def get_news_sentiment(self, symbol):
        try:
            url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={self.news_api_key}"
            response = requests.get(url).json()
            articles = response.get('articles', [])
            sia = SentimentIntensityAnalyzer()
            sentiments = [sia.polarity_scores(article['title'] + ' ' + (article['description'] or ''))['compound']
                          for article in articles[:5]]
            return np.mean(sentiments) if sentiments else 0
        except:
            return 0

    def train_model(self):
        symbol = self.symbol_entry.get().upper()
        if not symbol:
            self.status("Please enter a stock symbol", color='red')
            return

        self.data = self.fetch_data(symbol)
        if self.data is None or len(self.data) < 100:
            self.status("Insufficient data for training", color='red')
            return

        self.X, y = self.preprocess_data(self.data, symbol)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.X, y)

        # Calculate RMSE
        predictions = self.model.predict(self.X)
        rmse = np.sqrt(mean_squared_error(y, predictions))

        # Update results
        self.result_tree.delete(*self.result_tree.get_children())
        self.result_tree.insert('', 'end', values=('Model', 'Random Forest'))
        self.result_tree.insert('', 'end', values=('Training RMSE', f"{rmse:.2f}"))
        self.result_tree.insert('', 'end', values=('Last Close Price', f"{self.data['Close'].iloc[-1]:.2f}"))
        self.result_tree.insert('', 'end', values=('Feature Importance', self.get_feature_importance()))

        self.predict_btn.config(state='normal')
        self.status("Model trained successfully! Ready for prediction", color='green')
        self.update_visualization()

    def get_feature_importance(self):
        importances = self.model.feature_importances_
        features = self.X.columns
        top_features = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:3]
        return ', '.join([f"{feat}: {imp:.2f}" for feat, imp in top_features])

    def predict_tomorrow(self):
        if not self.model:
            self.status("Please train the model first", color='red')
            return

        prediction_date = self.get_next_trading_day()
        features = self.X.iloc[[-1]].copy()
        features['Sentiment'] = self.get_news_sentiment(self.symbol_entry.get().upper())

        prediction = self.model.predict(features)[0]
        last_close = self.data['Close'].iloc[-1]

        self.result_tree.insert('', 'end', values=('Prediction Date', prediction_date.strftime('%Y-%m-%d')))
        self.result_tree.insert('', 'end', values=('Tomorrow\'s Prediction', f"{prediction:.2f}"))
        self.result_tree.insert('', 'end', values=('Price Change', f"{(prediction - last_close):.2f}"))

        self.status("Prediction complete! Check results below", color='green')

    def get_next_trading_day(self):
        nyse = mcal.get_calendar('NYSE')
        next_day = datetime.now() + timedelta(days=1)
        while True:
            schedule = nyse.schedule(start_date=next_day, end_date=next_day)
            if not schedule.empty:
                return next_day
            next_day += timedelta(days=1)

    def update_visualization(self):
        self.ax.clear()
        self.ax.plot(self.data.index, self.data['Close'], label='Historical Price')
        self.ax.plot(self.data.index[:-1], self.model.predict(self.X), label='Model Predictions', alpha=0.7)
        self.ax.set_title('Price vs Predictions')
        self.ax.legend()
        self.canvas.draw()

    def buy_stock(self):
        symbol = self.portfolio_symbol.get().upper()
        try:
            shares = int(self.portfolio_shares.get())
            price = yf.Ticker(symbol).history(period='1d')['Close'].iloc[-1]
            cost = shares * price

            if self.portfolio['cash'] >= cost:
                self.portfolio['cash'] -= cost
                self.portfolio['holdings'][symbol] = self.portfolio['holdings'].get(symbol, 0) + shares
                self.update_portfolio_display()
                messagebox.showinfo("Success", f"Bought {shares} shares of {symbol}")
            else:
                messagebox.showwarning("Error", "Insufficient funds")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")

    def sell_stock(self):
        symbol = self.portfolio_symbol.get().upper()
        try:
            shares = int(self.portfolio_shares.get())
            if symbol in self.portfolio['holdings'] and self.portfolio['holdings'][symbol] >= shares:
                price = yf.Ticker(symbol).history(period='1d')['Close'].iloc[-1]
                self.portfolio['cash'] += shares * price
                self.portfolio['holdings'][symbol] -= shares
                if self.portfolio['holdings'][symbol] == 0:
                    del self.portfolio['holdings'][symbol]
                self.update_portfolio_display()
                messagebox.showinfo("Success", f"Sold {shares} shares of {symbol}")
            else:
                messagebox.showwarning("Error", "Not enough shares")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")

    def update_portfolio_display(self):
        self.portfolio_tree.delete(*self.portfolio_tree.get_children())
        self.portfolio_tree.insert('', 'end', values=('Cash', f"${self.portfolio['cash']:.2f}"))

        total_value = self.portfolio['cash']
        for symbol, shares in self.portfolio['holdings'].items():
            try:
                price = yf.Ticker(symbol).history(period='1d')['Close'].iloc[-1]
                value = shares * price
                total_value += value
                self.portfolio_tree.insert('', 'end', values=(symbol, f"{shares} shares (${value:.2f})"))
            except:
                self.portfolio_tree.insert('', 'end', values=(symbol, "Error fetching price"))

        self.portfolio_tree.insert('', 'end', values=('Total Value', f"${total_value:.2f}"))

    def status(self, message, color='black'):
        self.status_var.set(message)
        self.prediction_tab.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictorApp(root)
    root.mainloop()