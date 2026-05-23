#Colab code

"""
================================================================
NASDAQ HİSSE SENEDİ YÖN TAHMİNİ — V8 (FINAL)
================================================================
Proje   : Makine Öğrenmesi ile NASDAQ Hisselerinde Yön Tahmini
Yazar   : Can Aslan | 02230201021
Ders    : Yapay Zekâ — İnönü Üniversitesi

V7 → V8 Tek Düzeltme:
  [FIX] XGBoost eval_set data leakage giderildi.
        Önceki kodda eval_set=(X_test_scaled, y_test) verilmişti
        yani model eğitim sırasında test verisini görüyordu.
        Düzeltme: train içinden %10 validation ayrılıp kullanıldı.
================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)

# ──────────────────────────────────────────────────────────────
# 0. AYARLAR
# ──────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "NFLX", "AMD", "INTC",
    "CSCO", "PEP", "AVGO", "TXN", "QCOM",
    "ADBE", "CRM", "AMAT", "MU", "PYPL"
]
PERIOD      = "10y"
TEST_SIZE   = 0.20
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────
# 1. PİYASA VERİSİNİ ÇEK (SPY, VIX, QQQ)
# ──────────────────────────────────────────────────────────────
print("=" * 62)
print("V8 (Final) — Piyasa Verisi Çekiliyor...")
print("=" * 62)

def to_date_index(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

print("[*] SPY, VIX, QQQ çekiliyor...")

spy_raw = to_date_index(yf.Ticker("SPY").history(period=PERIOD))
vix_raw = to_date_index(yf.Ticker("^VIX").history(period=PERIOD))
qqq_raw = to_date_index(yf.Ticker("QQQ").history(period=PERIOD))

market = pd.DataFrame({
    'SPY_Return' : spy_raw['Close'].pct_change(),
    'QQQ_Return' : qqq_raw['Close'].pct_change(),
    'VIX'        : vix_raw['Close'],
    'VIX_Change' : vix_raw['Close'].pct_change(),
}).dropna()

print(f"[+] Piyasa verisi hazır: {len(market)} satır "
      f"({market.index[0].date()} – {market.index[-1].date()})\n")

# ──────────────────────────────────────────────────────────────
# 2. YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────────────────────

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(window).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_features(df, market, ticker_name=""):
    df    = to_date_index(df)
    close = df['Close']

    sma_10 = close.rolling(window=10).mean()
    sma_50 = close.rolling(window=50).mean()
    df['SMA_Ratio']      = sma_10 / sma_50
    df['Price_vs_SMA10'] = (close / sma_10) - 1
    df['Price_vs_SMA50'] = (close / sma_50) - 1
    df['High_Low_Range'] = (df['High'] - df['Low']) / close
    df['Volume_Change']  = df['Volume'].pct_change()
    df['RSI']            = calculate_rsi(close)
    df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
    df['RSI_Oversold']   = (df['RSI'] < 30).astype(int)

    exp1           = close.ewm(span=12, adjust=False).mean()
    exp2           = close.ewm(span=26, adjust=False).mean()
    macd_raw       = exp1 - exp2
    signal_raw     = macd_raw.ewm(span=9, adjust=False).mean()
    df['MACD_Norm']        = macd_raw   / close
    df['Signal_Norm']      = signal_raw / close
    df['MACD_Signal_Flag'] = (macd_raw > signal_raw).astype(int)

    df['Daily_Return']  = close.pct_change()
    df['Return_5d']     = close.pct_change(5)
    df['Return_10d']    = close.pct_change(10)
    df['Volatility_10'] = df['Daily_Return'].rolling(10).std()

    df['Target'] = np.where(close.shift(-1) > close * 1.001, 1, 0)

    df = df.merge(market, left_index=True, right_index=True, how='left')

    df['Relative_to_QQQ'] = df['Daily_Return'] - df['QQQ_Return']
    df['VIX_High']         = (df['VIX'] > 25).astype(int)
    df['VIX_Low']          = (df['VIX'] < 15).astype(int)

    df['Ticker'] = ticker_name
    return df


FEATURE_COLS = [
    'SMA_Ratio', 'Price_vs_SMA10', 'Price_vs_SMA50',
    'High_Low_Range', 'Volume_Change',
    'RSI', 'RSI_Overbought', 'RSI_Oversold',
    'MACD_Norm', 'Signal_Norm', 'MACD_Signal_Flag',
    'Daily_Return', 'Return_5d', 'Return_10d', 'Volatility_10',
    'SPY_Return', 'QQQ_Return',
    'VIX', 'VIX_Change',
    'Relative_to_QQQ',
    'VIX_High', 'VIX_Low',
]

# ──────────────────────────────────────────────────────────────
# 3. VERİ TOPLAMA
# ──────────────────────────────────────────────────────────────
print("[*] Hisse verileri çekiliyor...")

all_frames = []
basarili, basarisiz = 0, 0

for ticker in TICKERS:
    try:
        df = yf.Ticker(ticker).history(period=PERIOD)
        df = df.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
        df = build_features(df, market, ticker_name=ticker)
        df.dropna(inplace=True)

        if len(df) == 0:
            print(f"  [!] {ticker:5s} — 0 satır, atlandı.")
            basarisiz += 1
            continue

        all_frames.append(df)
        basarili += 1
        print(f"  [+] {ticker:5s} — {len(df):4d} satır, {len(FEATURE_COLS)} feature")
    except Exception as e:
        basarisiz += 1
        print(f"  [-] {ticker} atlandı: {e}")

print(f"\n  Başarılı: {basarili} hisse | Atlandı: {basarisiz} hisse")

if len(all_frames) == 0:
    raise RuntimeError("Hiçbir hisse işlenemedi!")

# ──────────────────────────────────────────────────────────────
# 4. BİRLEŞTİRME VE ÖN İŞLEME
# ──────────────────────────────────────────────────────────────
all_data = pd.concat(all_frames).sort_index()

X = all_data[FEATURE_COLS]
y = all_data['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False
)

print(f"\n  Eğitim seti    : {len(X_train):6,} satır")
print(f"  Test seti      : {len(X_test):6,} satır")
print(f"  Feature sayısı : {len(FEATURE_COLS)}")
print(f"  Sınıf dengesi  : %{y_train.mean()*100:.1f} pozitif — "
      f"{'DENGELI ✓' if 0.45 < y_train.mean() < 0.55 else 'DENGESİZ !'}\n")

scaler         = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ──────────────────────────────────────────────────────────────
# 5. MODEL EĞİTİMİ
# ──────────────────────────────────────────────────────────────
print("=" * 62)
print("Model Eğitimi Başlıyor...")
print("=" * 62)

tscv = TimeSeriesSplit(n_splits=5)

# ── Model 1: Lojistik Regresyon ───────────────────────────────
t0 = time.time()
lr_model = LogisticRegression(
    class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED
)
lr_model.fit(X_train_scaled, y_train)
print(f"[+] Lojistik Regresyon tamamlandı. ({time.time()-t0:.1f} sn)")

# ── Model 2: MLP ──────────────────────────────────────────────
t0 = time.time()
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32), activation='relu',
    max_iter=1000, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=20,
    random_state=RANDOM_SEED
)
mlp_model.fit(X_train_scaled, y_train)
print(f"[+] MLP tamamlandı. ({time.time()-t0:.1f} sn) | {mlp_model.n_iter_} iterasyon")

# ── Model 3: XGBoost ──────────────────────────────────────────
val_size      = int(len(X_train_scaled) * 0.10)
X_xgb_train   = X_train_scaled[:-val_size]
y_xgb_train   = y_train.values[:-val_size]
X_xgb_val     = X_train_scaled[-val_size:]
y_xgb_val     = y_train.values[-val_size:]

t0 = time.time()
xgb_model = XGBClassifier(
    n_estimators          = 300,
    max_depth             = 4,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    early_stopping_rounds = 20,
    eval_metric           = 'logloss',
    random_state          = RANDOM_SEED,
    n_jobs                = -1,
)
xgb_model.fit(
    X_xgb_train, y_xgb_train,
    eval_set=[(X_xgb_val, y_xgb_val)],   # test verisi görmüyor
    verbose=False
)
best_iter = xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else 'N/A'
print(f"[+] XGBoost tamamlandı. ({time.time()-t0:.1f} sn) | En iyi ağaç: {best_iter}")

# ── Model 4: SVM ──────────────────────────────────────────────
t0 = time.time()
svm_grid = GridSearchCV(
    SVC(kernel='rbf', probability=True,
        class_weight='balanced', random_state=RANDOM_SEED),
    param_grid={'C': [0.1, 1, 10]},
    cv=tscv, scoring='roc_auc', n_jobs=-1, verbose=0
)
svm_grid.fit(X_train_scaled, y_train)
best_svm = svm_grid.best_estimator_
print(f"[+] SVM tamamlandı. "
      f"En iyi C={svm_grid.best_params_['C']} | "
      f"CV ROC-AUC={svm_grid.best_score_:.4f} "
      f"({time.time()-t0:.1f} sn)")

# ──────────────────────────────────────────────────────────────
# 6. PERFORMANS DEĞERLENDİRME
# ──────────────────────────────────────────────────────────────

def evaluate_model(model, X_te, y_te, model_name):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    roc  = roc_auc_score(y_te, y_prob)
    cm   = confusion_matrix(y_te, y_pred)

    print(f"\n{'─' * 50}")
    print(f"  {model_name}")
    print(f"{'─' * 50}")
    print(f"  Accuracy  (Doğruluk)  : %{acc*100:.2f}")
    print(f"  Precision (Kesinlik)  : %{prec*100:.2f}")
    print(f"  Recall    (Duyarlılık): %{rec*100:.2f}")
    print(f"  F1-Score              : %{f1*100:.2f}")
    print(f"  ROC-AUC               : {roc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    return y_prob


print("\n" + "=" * 62)
print("PERFORMANS SONUÇLARI — Test Seti")
print("=" * 62)

models = {
    "Lojistik Regresyon (Baseline)": lr_model,
    "MLP (128-64-32, Early Stop)"  : mlp_model,
    "XGBoost (300 ağaç)"           : xgb_model,
    "SVM — RBF Kernel (Optimize)"  : best_svm,
}

probabilities = {}
for name, model in models.items():
    prob = evaluate_model(model, X_test_scaled, y_test, name)
    probabilities[name] = prob

print(f"\n{'─' * 50}")

# ──────────────────────────────────────────────────────────────
# 7. ROC EĞRİSİ
# ──────────────────────────────────────────────────────────────
print("\n[*] ROC eğrisi oluşturuluyor...")

colors = ['steelblue', 'seagreen', 'crimson', 'darkorange']
plt.figure(figsize=(8, 6))

for (name, prob), color in zip(probabilities.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val      = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f"{name} (AUC = {auc_val:.4f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Rastgele Tahmin (AUC = 0.50)')
plt.xlabel('Yanlış Pozitif Oranı (FPR)', fontsize=12)
plt.ylabel('Doğru Pozitif Oranı (TPR)', fontsize=12)
plt.title('ROC Eğrisi — NASDAQ Yön Tahmini (V8 Final)', fontsize=13)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_v8.png', dpi=150)
print("[+] ROC eğrisi 'roc_curve_v8.png' olarak kaydedildi.")

# ──────────────────────────────────────────────────────────────
# 8. MODEL KAYDETME
# ──────────────────────────────────────────────────────────────
joblib.dump(lr_model,  'lr_model_v8.joblib')
joblib.dump(mlp_model, 'mlp_model_v8.joblib')
joblib.dump(xgb_model, 'xgb_model_v8.joblib')
joblib.dump(best_svm,  'svm_model_v8.joblib')
joblib.dump(scaler,    'scaler_v8.joblib')

print("\n[+] Modeller kaydedildi:")
print("    lr_model_v8.joblib")
print("    mlp_model_v8.joblib")
print("    xgb_model_v8.joblib")
print("    svm_model_v8.joblib")
print("    scaler_v8.joblib")
print("    roc_curve_v8.png")
print("\n" + "=" * 62)
print("V8 Final Pipeline başarıyla tamamlandı.")
print("=" * 62)