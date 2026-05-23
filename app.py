"""
================================================================
NASDAQ YÖN TAHMİN UYGULAMASI — Streamlit Arayüzü
================================================================
Yazar  : Can Aslan | 02230201021
Ders   : Yapay Zekâ — İnönü Üniversitesi
Çalıştır: streamlit run app.py
================================================================
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# ──────────────────────────────────────────────────────────────
# 0. SAYFA AYARLARI
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NASDAQ Yön Tahmini",
    page_icon="📈",
    layout="wide"
)

# ──────────────────────────────────────────────────────────────
# 1. MODEL YÜKLEME
# ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_models():
    lr  = joblib.load(os.path.join(MODEL_DIR, "lr_model_v8.joblib"))
    mlp = joblib.load(os.path.join(MODEL_DIR, "mlp_model_v8.joblib"))
    xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_model_v8.joblib"))
    svm = joblib.load(os.path.join(MODEL_DIR, "svm_model_v8.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_v8.joblib"))
    return lr, mlp, xgb, svm, scaler

lr_model, mlp_model, xgb_model, svm_model, scaler = load_models()

MODELS = {
    "Lojistik Regresyon": lr_model,
    "MLP (Sinir Ağı)":    mlp_model,
    "XGBoost":            xgb_model,
    "SVM":                svm_model,
}

# ──────────────────────────────────────────────────────────────
# 2. FEATURE MÜHENDİSLİĞİ FONKSİYONLARI
# ──────────────────────────────────────────────────────────────

def to_date_index(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(window).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600)
def get_market_data():
    spy = to_date_index(yf.Ticker("SPY").history(period="3mo"))
    vix = to_date_index(yf.Ticker("^VIX").history(period="3mo"))
    qqq = to_date_index(yf.Ticker("QQQ").history(period="3mo"))
    market = pd.DataFrame({
        'SPY_Return': spy['Close'].pct_change(),
        'QQQ_Return': qqq['Close'].pct_change(),
        'VIX':        vix['Close'],
        'VIX_Change': vix['Close'].pct_change(),
    }).dropna()
    return market

def build_features(ticker_symbol, market):
    df = yf.Ticker(ticker_symbol).history(period="1y")
    if df.empty:
        return None, None
    df = to_date_index(df)
    df = df.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
    close = df['Close']

    sma_10 = close.rolling(10).mean()
    sma_50 = close.rolling(50).mean()
    df['SMA_Ratio']        = sma_10 / sma_50
    df['Price_vs_SMA10']   = (close / sma_10) - 1
    df['Price_vs_SMA50']   = (close / sma_50) - 1
    df['High_Low_Range']   = (df['High'] - df['Low']) / close
    df['Volume_Change']    = df['Volume'].pct_change()
    df['RSI']              = calculate_rsi(close)
    df['RSI_Overbought']   = (df['RSI'] > 70).astype(int)
    df['RSI_Oversold']     = (df['RSI'] < 30).astype(int)
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd_raw = exp1 - exp2
    signal_raw = macd_raw.ewm(span=9, adjust=False).mean()
    df['MACD_Norm']        = macd_raw / close
    df['Signal_Norm']      = signal_raw / close
    df['MACD_Signal_Flag'] = (macd_raw > signal_raw).astype(int)
    df['Daily_Return']     = close.pct_change()
    df['Return_5d']        = close.pct_change(5)
    df['Return_10d']       = close.pct_change(10)
    df['Volatility_10']    = df['Daily_Return'].rolling(10).std()
    df = df.merge(market, left_index=True, right_index=True, how='left')
    df['Relative_to_QQQ']  = df['Daily_Return'] - df['QQQ_Return']
    df['VIX_High']          = (df['VIX'] > 25).astype(int)
    df['VIX_Low']           = (df['VIX'] < 15).astype(int)
    df.dropna(inplace=True)

    if df.empty:
        return None, None

    feature_cols = [
        'SMA_Ratio', 'Price_vs_SMA10', 'Price_vs_SMA50',
        'High_Low_Range', 'Volume_Change',
        'RSI', 'RSI_Overbought', 'RSI_Oversold',
        'MACD_Norm', 'Signal_Norm', 'MACD_Signal_Flag',
        'Daily_Return', 'Return_5d', 'Return_10d', 'Volatility_10',
        'SPY_Return', 'QQQ_Return', 'VIX', 'VIX_Change',
        'Relative_to_QQQ', 'VIX_High', 'VIX_Low',
    ]

    last_row = df[feature_cols].iloc[[-1]]
    last_row_scaled = scaler.transform(last_row)
    return last_row_scaled, df

# ──────────────────────────────────────────────────────────────
# 3. BAŞLIK
# ──────────────────────────────────────────────────────────────
st.title("📈 NASDAQ Hisse Senedi Yön Tahmin Sistemi")
st.caption("Can Aslan | 02230201021 | İnönü Üniversitesi — Yapay Zekâ Dersi")
st.markdown("---")

# ──────────────────────────────────────────────────────────────
# 4. TABLAR
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔮 Yarınki Yön Tahmini",
    "📊 Model Performansı",
    "ℹ️ Proje Hakkında"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — TAHMİN
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Hisse Senedi Yön Tahmini")
    st.write("Bir NASDAQ hissesi seçin veya sembol girin. Model, yarınki kapanış yönünü tahmin edecek.")

    col1, col2 = st.columns([2, 1])

    with col1:
        populer = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
                   "META", "TSLA", "AMD", "NFLX", "INTC"]
        secilen = st.selectbox("Popüler NASDAQ Hisseleri", populer)
        manuel  = st.text_input("veya kendiniz yazın (örn: PYPL, CRM, AVGO)", "").upper().strip()
        ticker  = manuel if manuel else secilen

    with col2:
        st.metric("Seçili Hisse", ticker)

    tahmin_btn = st.button("🔍 Tahmin Et", type="primary", use_container_width=True)

    if tahmin_btn:
        with st.spinner(f"{ticker} verisi çekiliyor ve analiz ediliyor..."):
            try:
                market = get_market_data()
                X_son, df_hisse = build_features(ticker, market)

                if X_son is None:
                    st.error(f"'{ticker}' için yeterli veri bulunamadı. Sembolü kontrol edin.")
                else:
                    # Güncel fiyat bilgisi
                    son_fiyat   = df_hisse['Close'].iloc[-1]
                    onceki_fiyat = df_hisse['Close'].iloc[-2]
                    gunluk_deg  = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100

                    st.markdown("### 💰 Güncel Fiyat Bilgisi")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Son Kapanış", f"${son_fiyat:.2f}",
                              f"{gunluk_deg:+.2f}% bugün")
                    m2.metric("RSI (14)", f"{df_hisse['RSI'].iloc[-1]:.1f}")
                    m3.metric("Volatilite (10g)",
                              f"{df_hisse['Volatility_10'].iloc[-1]*100:.2f}%")

                    st.markdown("### 🤖 Model Tahminleri")

                    cols = st.columns(4)
                    for idx, (model_adi, model) in enumerate(MODELS.items()):
                        prob      = model.predict_proba(X_son)[0][1]
                        tahmin    = int(prob >= 0.5)
                        yukselis  = prob
                        dusus     = 1 - prob

                        with cols[idx]:
                            st.markdown(f"**{model_adi}**")
                            if tahmin == 1:
                                st.success(f"📈 YÜKSELİR")
                            else:
                                st.error(f"📉 DÜŞER")
                            st.progress(float(yukselis))
                            st.caption(f"↑ {yukselis*100:.1f}%  |  ↓ {dusus*100:.1f}%")

                    # Oylama
                    st.markdown("---")
                    oylar = []
                    for model in MODELS.values():
                        p = model.predict_proba(X_son)[0][1]
                        oylar.append(1 if p >= 0.5 else 0)

                    yukselis_oyu = sum(oylar)
                    dusus_oyu   = 4 - yukselis_oyu

                    st.markdown("### 🗳️ Model Oylama Sonucu")
                    oy_col1, oy_col2 = st.columns(2)
                    oy_col1.metric("📈 Yükseliş Oyu", f"{yukselis_oyu} / 4")
                    oy_col2.metric("📉 Düşüş Oyu",   f"{dusus_oyu} / 4")

        

                    st.info(
                        "⚠️ **Sorumluluk Reddi:** Bu sistem akademik amaçlı geliştirilmiştir. "
                        "Kesin finansal tavsiye niteliği taşımaz. "
                        "Yatırım kararlarınızı bir uzmana danışarak verin."
                    )

            except Exception as e:
                st.error(f"Hata oluştu: {e}")

# ══════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANSI
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performans Metrikleri")
    st.write("V8 eğitiminden elde edilen test seti sonuçları.")

    # Metrik tablosu
    performans = {
        "Model": [
            "Lojistik Regresyon",
            "MLP (128-64-32)",
            "XGBoost (300 ağaç)",
            "SVM — RBF Kernel"
        ],
        "Accuracy": ["%50.54", "%50.17", "%50.14", "%49.92"],
        "Precision": ["%50.57", "%50.11", "%50.06", "%49.85"],
        "Recall": ["%41.74", "%45.24", "%61.99", "%51.55"],
        "F1-Score": ["%45.73", "%47.55", "%55.39", "%50.69"],
        "ROC-AUC": ["0.5100", "0.5089", "0.5030", "0.5033"],
    }

    df_perf = pd.DataFrame(performans)
    st.dataframe(df_perf, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ROC Eğrisi
    st.subheader("📉 ROC Eğrisi")
    roc_path = os.path.join(MODEL_DIR, "roc_curve_v8.png")
    if os.path.exists(roc_path):
        st.image(roc_path, use_container_width=True)
    else:
        st.warning("roc_curve_v8.png bulunamadı. models/ klasörüne kopyalayın.")

    st.markdown("---")

    # Confusion Matrix
    st.subheader("🔲 Confusion Matrix")
    st.write("Test seti üzerindeki tahmin dağılımları:")

    cm_data = {
        "Lojistik Regresyon": np.array([[2928, 2009], [2868, 2055]]),
        "MLP (128-64-32)":    np.array([[2720, 2217], [2696, 2227]]),
        "XGBoost":            np.array([[1892, 3045], [1871, 3052]]),
        "SVM — RBF":          np.array([[2384, 2553], [2385, 2538]]),
    }

    cm_cols = st.columns(2)
    for idx, (model_adi, cm) in enumerate(cm_data.items()):
        with cm_cols[idx % 2]:
            fig, ax = plt.subplots(figsize=(4, 3))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Düşer (0)", "Yükselir (1)"]
            )
            disp.plot(ax=ax, colorbar=False, cmap='Blues')
            ax.set_title(model_adi, fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 3 — HAKKINDA
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Proje Hakkında")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📌 Proje Bilgisi
        | | |
        |---|---|
        | **Proje Adı** | NASDAQ Hisselerinde Yön Tahmini |
        | **Yazar** | Can Aslan |
        | **Öğrenci No** | 02230201021 |
        | **Üniversite** | İnönü Üniversitesi |
        | **Ders** | Yapay Zekâ |
        | **Model Versiyonu** | V8 Final |

        ### 🎯 Problem Tanımı
        Bir NASDAQ hisse senedinin ertesi gün kapanış fiyatının
        bugüne göre **yükseliş mi düşüş mü** göstereceğinin
        tahmin edilmesi. İkili sınıflandırma problemi.

        - **1** → Yükselir (Close[t+1] > Close[t] × 1.001)
        - **0** → Düşer veya aynı kalır
        """)

    with col2:
        st.markdown("""
        ### 🔧 Kullanılan Teknolojiler
        | | |
        |---|---|
        | **Dil** | Python 3.12 |
        | **Veri** | Yahoo Finance (yfinance) |
        | **ML** | scikit-learn, XGBoost |
        | **Arayüz** | Streamlit |
        | **Veri Dönemi** | 10 yıl (2016–2026) |
        | **Hisse Sayısı** | 20 NASDAQ hissesi |
        | **Feature Sayısı** | 22 |

        ### 📊 Veri Kaynakları
        - **Hisse verileri:** Yahoo Finance (OHLCV)
        - **Piyasa bağlamı:** SPY, QQQ, VIX
        - **Teknik indikatörler:** RSI, MACD, SMA, Bollinger
        """)

    st.markdown("---")
    st.markdown("""
    ### 🧠 Kullanılan Modeller
    | Model | Açıklama | Eğitim Süresi |
    |---|---|---|
    | Lojistik Regresyon | Baseline model | 0.3 sn |
    | MLP (128-64-32) | Çok katmanlı sinir ağı | ~40 sn |
    | XGBoost | Gradient boosting, 300 ağaç | 0.4 sn |
    | SVM (RBF) | Destek vektör makinesi | ~75 dk |

    ### ⚠️ Sorumluluk Reddi
    Bu sistem yalnızca **akademik ve eğitim amaçlı** geliştirilmiştir.
    Üretilen tahminler kesin finansal tavsiye niteliği taşımamakta olup
    gerçek yatırım kararlarında kullanılmamalıdır.
    Yatırım kararları için lisanslı finansal danışmanlara başvurunuz.
    """)