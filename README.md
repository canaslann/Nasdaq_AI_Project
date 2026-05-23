# 📈 NASDAQ Stock Direction Prediction using Machine Learning

> **Course Project** | Artificial Intelligence | İnönü University  
> **Author:** Can Aslan | **Student ID:** 02230201021

---

## 🎯 Project Overview

This project tackles the problem of predicting the **next-day price direction** (up or down) of NASDAQ-listed stocks using machine learning classification algorithms combined with technical indicator feature engineering and market context data.

Instead of predicting exact prices (regression), the problem is framed as a **binary classification task**:

- **1** → Tomorrow's closing price will be **higher** than today (by more than 0.1%)
- **0** → Tomorrow's closing price will be **lower** or stay the same

---

## 🗂️ Project Structure

```
nasdaq_project/
├── app.py                  # Streamlit web application
├── NASDAQ_Baslat.bat       # Windows one-click launcher
├── nasdaq_v8_final.py      # Model training pipeline (V8 Final)
└── models/
    ├── lr_model_v8.joblib  # Logistic Regression
    ├── mlp_model_v8.joblib # Multi-Layer Perceptron
    ├── xgb_model_v8.joblib # XGBoost
    ├── svm_model_v8.joblib # Support Vector Machine
    ├── scaler_v8.joblib    # RobustScaler
    └── roc_curve_v8.png    # ROC curve comparison plot
```

---

## 🧠 Models Used

| Model | Description | Training Time |
|---|---|---|
| Logistic Regression | Baseline model | ~0.3 sec |
| MLP (128-64-32) | Multi-layer neural network with early stopping | ~40 sec |
| XGBoost | Gradient boosting, 300 trees | ~0.4 sec |
| SVM (RBF Kernel) | Support Vector Machine with Grid Search | ~75 min |

---

## 📊 Features (22 total)

### Technical Indicators (15)
| Feature | Description |
|---|---|
| `SMA_Ratio` | SMA-10 / SMA-50 ratio |
| `Price_vs_SMA10` | Price deviation from SMA-10 |
| `Price_vs_SMA50` | Price deviation from SMA-50 |
| `High_Low_Range` | Normalized daily range |
| `Volume_Change` | Daily volume % change |
| `RSI` | Relative Strength Index (14-day) |
| `RSI_Overbought` | Binary flag: RSI > 70 |
| `RSI_Oversold` | Binary flag: RSI < 30 |
| `MACD_Norm` | Normalized MACD line |
| `Signal_Norm` | Normalized signal line |
| `MACD_Signal_Flag` | MACD crossover binary flag |
| `Daily_Return` | Daily price % change |
| `Return_5d` | 5-day cumulative return |
| `Return_10d` | 10-day cumulative return |
| `Volatility_10` | 10-day rolling volatility |

### Market Context (7)
| Feature | Description |
|---|---|
| `SPY_Return` | S&P 500 ETF daily return |
| `QQQ_Return` | NASDAQ-100 ETF daily return |
| `VIX` | CBOE Volatility Index level |
| `VIX_Change` | VIX daily % change |
| `Relative_to_QQQ` | Stock return minus QQQ return |
| `VIX_High` | Binary flag: VIX > 25 (fear) |
| `VIX_Low` | Binary flag: VIX < 15 (optimism) |

---

## 📈 Results

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 50.54% | 45.73% | 0.5100 |
| MLP (128-64-32) | 50.17% | 47.55% | 0.5089 |
| XGBoost | 50.14% | **55.39%** | 0.5030 |
| SVM — RBF | 49.92% | 50.69% | 0.5033 |

### 📌 Interpretation

The ROC-AUC values hovering around **0.51** are consistent with the **Efficient Market Hypothesis (EMH)**, which suggests that publicly available technical data cannot reliably predict future price movements beyond chance. Most studies in the literature claiming 60%+ accuracy contain methodological flaws such as **data leakage**. This pipeline is methodologically clean:

- ✅ Chronological train/test split (no shuffling)
- ✅ Scaler fitted only on training data
- ✅ Per-ticker split before concatenation
- ✅ TimeSeriesSplit for cross-validation
- ✅ No arbitrary threshold manipulation

---

## 🗄️ Dataset

- **Source:** Yahoo Finance via `yfinance` Python library
- **Tickers:** 20 major NASDAQ stocks (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, NFLX, AMD, INTC, CSCO, PEP, AVGO, TXN, QCOM, ADBE, CRM, AMAT, MU, PYPL)
- **Market Context:** SPY, QQQ, ^VIX
- **Period:** 10 years (2016–2026)
- **Total Samples:** ~49,000 rows
- **Train / Test Split:** 80% / 20% (chronological)
- **Class Balance:** ~50.1% positive — balanced ✓

---

## 🖥️ Web Application

A **Streamlit** web application is included with 3 tabs:

### Tab 1 — 🔮 Direction Prediction
- Select or type any NASDAQ ticker symbol
- Get real-time predictions from all 4 models
- View confidence scores and majority vote consensus

### Tab 2 — 📊 Model Performance
- Performance metrics table
- ROC curve comparison chart
- Confusion matrices for all 4 models

### Tab 3 — ℹ️ About
- Project details, methodology, and disclaimer

---

## 🚀 Installation & Usage

### Requirements

```bash
pip install streamlit yfinance xgboost scikit-learn joblib matplotlib
```

### Run the App

**Option 1 — Windows (easiest):**
Double-click `NASDAQ_Baslat.bat`

**Option 2 — Terminal:**
```bash
cd nasdaq_project
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

### Retrain Models

```bash
python nasdaq_v8_final.py
```

> ⚠️ SVM training takes approximately 60–90 minutes on CPU.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| yfinance | Market data fetching |
| pandas, numpy | Data processing |
| scikit-learn | LR, MLP, SVM, preprocessing, metrics |
| XGBoost | Gradient boosting classifier |
| Streamlit | Web application framework |
| matplotlib | ROC curve visualization |
| joblib | Model serialization |

---

## ⚠️ Disclaimer

This project was developed for **academic and educational purposes only**. The predictions generated by this system do not constitute financial advice and should not be used for real investment decisions. Always consult a licensed financial advisor before making investment decisions.

---

## 📚 References

1. Usmani, M., Adil, S. H., Raza, K., & Ali, S. S. A. (2016). Stock Market Prediction Using Machine Learning Techniques. *ICCOINS 2016*, IEEE.
2. Parmar, I., et al. (2018). Stock Market Prediction Using Machine Learning. *ICSCCC 2018*, IEEE.
3. Hegazy, O., Soliman, O. S., & Salam, M. A. (2013). A Machine Learning Model for Stock Market Prediction. *IJCST*, 4(12).
4. Reddy, V. K. S. (2018). Stock Market Prediction Using Machine Learning. *IRJET*, 5(10).
5. Strader, T. J., et al. (2020). Machine Learning Stock Market Prediction Studies: Review and Research Directions. *JITIM*, 28(4).
6. Rouf, N., et al. (2021). Stock Market Prediction Using Machine Learning Techniques: A Decade Survey. *Electronics*, 10(21), 2717.

---

<div align="center">
  <b>İnönü University — Computer Engineering</b><br>
  Artificial Intelligence Course Project | 2026
</div>
