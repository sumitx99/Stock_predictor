# 📈 Enhanced Market Predictor 🧠💹

An AI-powered desktop application for predicting stock prices, analyzing news sentiment, visualizing trends, and simulating portfolio management — all wrapped in an interactive Tkinter GUI.

<div align="center">
  
![TRAFFIC GIF](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDBpMmtyeDhpaTk2bWxtdmNpaWNiOHpuODlxeHNybWZ6MWVnNGtpdiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/SaX384PjtDl2U/giphy.gif)

</div>
---

## 🚀 Features

- 🔍 **Stock Prediction**
  - Fetches historical stock data using `yfinance`
  - Calculates technical indicators (MA, RSI, MACD, Bollinger Bands)
  - Integrates real-time news sentiment analysis (NewsAPI + VADER)
  - Predicts next-day closing price using Random Forest Regressor
  - Evaluates prediction accuracy with RMSE

- 📊 **Data Visualization**
  - Plots actual vs predicted stock prices
  - Interactive charts via `matplotlib`

- 💼 **Portfolio Management**
  - Simulates stock buying/selling
  - Tracks holdings, cash balance, and total portfolio value
  - Fetches live prices for evaluation

- 🖼️ **User Interface**
  - Clean, responsive UI built using `Tkinter` + `ttkbootstrap`
  - Tabbed navigation for Predictions, Visualization, and Portfolio

---

## 📦 Tech Stack

| Tool/Library             | Purpose                          |
|--------------------------|----------------------------------|
| `Tkinter` + `ttkbootstrap` | GUI and styled widgets         |
| `yfinance`               | Stock data fetching              |
| `NewsAPI`                | Real-time news headlines         |
| `nltk.sentiment.vader`   | News sentiment analysis          |
| `scikit-learn`           | Machine learning model (Random Forest) |
| `matplotlib`             | Graph plotting                   |
| `pandas_market_calendars`| Next trading day detection       |
| `pandas`, `numpy`        | Data manipulation                |

---

## 📷 Screenshots

<details>
<summary>Prediction Tab</summary>

![Prediction Tab](screenshots/prediction_tab.png)
</details>

<details>
<summary>Visualization Tab</summary>

![Visualization Tab](screenshots/visualization_tab.png)
</details>

<details>
<summary>Portfolio Tab</summary>

![Portfolio Tab](screenshots/portfolio_tab.png)
</details>

---
