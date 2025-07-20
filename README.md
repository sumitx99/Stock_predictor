# 📈 Enhanced Market Predictor 🧠💹

An AI-powered desktop application for predicting stock prices, analyzing news sentiment, visualizing trends, and simulating portfolio management — all wrapped in an interactive Tkinter GUI.

<div align="center">
  
![TRAFFIC GIF](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDBpMmtyeDhpaTk2bWxtdmNpaWNiOHpuODlxeHNybWZ6MWVnNGtpdiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/SaX384PjtDl2U/giphy.gif)

</div>

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

![Prediction Tab](https://github.com/user-attachments/assets/c9c11511-de63-497b-9bf3-d66023caaba8)


</details>

<details>
<summary>Visualization Tab</summary>

![Visualization Tab](https://github.com/user-attachments/assets/a2a786ae-f37c-42d3-87ab-4c17cf0490fc)

</details>

<details>
<summary>Portfolio Tab</summary>

![Portfolio Tab](https://github.com/user-attachments/assets/8ae6d2ae-d8da-4e77-a092-ca7ad2edc770)

</details>


---
