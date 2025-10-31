# app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="Model Reveal XAI", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
    font-family: 'Segoe UI', sans-serif;
}
.card {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
button {
    background-color: #0072C6;
    color: white;
    border-radius: 5px;
    padding: 8px 16px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.title("💡 Model Reveal: Explainable AI Dashboard")
st.markdown("Yüklediğiniz veri ile modeli eğitin ve SHAP ile karar mekanizmasını görün.")

# --- SESSION STATE ---
if 'stage' not in st.session_state:
    st.session_state.stage = 'upload_train'
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'le' not in st.session_state:
    st.session_state.le = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'train_df' not in st.session_state:
    st.session_state.train_df = None
if 'test_df' not in st.session_state:
    st.session_state.test_df = None

MODEL_OPTIONS = {
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "XGBoost": XGBClassifier
}

# --- Fonksiyonlar ---
@st.cache_data
def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata: {e}")
        return None

def preprocess(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        st.session_state.le = le
    else:
        st.session_state.le = None
    return X, y

def train_model(X, y, model_class):
    model = model_class(random_state=42)
    if model_class == RandomForestClassifier or model_class == XGBClassifier:
        model = model_class(random_state=42, n_estimators=100)
    model.fit(X, y)
    return model

def plot_shap_summary(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    fig, ax = plt.subplots(figsize=(10,6))
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig)
    plt.close(fig)
    return explainer, shap_values

def plot_shap_force(explainer, shap_values, X):
    if isinstance(shap_values, list):
        shap_val = shap_values[0][0]
    else:
        shap_val = shap_values[0]
    shap.initjs()
    st.subheader("Force Plot (İlk Örnek)")
    st_shap = st.pyplot
    shap.force_plot(explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                    shap_val, X.iloc[0,:], matplotlib=True)

# --- Aşamalar ---
# 1️⃣ Eğitim Verisi Yükle
if st.session_state.stage == 'upload_train':
    st.header("1️⃣ Eğitim Verisi Yükleyin (.csv)")
    uploaded_train_file = st.file_uploader("Eğitim CSV", type="csv")
    if uploaded_train_file is not None:
        df = load_csv(uploaded_train_file)
        if df is not None:
            st.session_state.train_df = df
            st.success("Eğitim verisi yüklendi!")
            st.dataframe(df.head())
            st.session_state.stage = 'select_target'
            st.experimental_rerun()

# 2️⃣ Target seçimi
if st.session_state.stage == 'select_target':
    st.header("2️⃣ Target Sütunu Seçin")
    target_column = st.selectbox("Target Sütunu", st.session_state.train_df.columns)
    if st.button("Onayla ve Model Seç"):
        st.session_state.target_column = target_column
        st.session_state.stage = 'select_model'
        st.experimental_rerun()

# 3️⃣ Model seçimi ve eğitimi
if st.session_state.stage == 'select_model':
    st.header("3️⃣ Model Seçin ve Eğitin")
    col1, col2, col3 = st.columns(3)
    for idx, (name, cls) in enumerate(MODEL_OPTIONS.items()):
        with [col1, col2, col3][idx]:
            if st.button(f"{name} Eğit"):
                st.session_state.model_type = name
                X, y = preprocess(st.session_state.train_df, st.session_state.target_column)
                st.session_state.X_train = X
                st.session_state.y_train = y
                st.session_state.model = train_model(X, y, cls)
                st.success(f"{name} başarıyla eğitildi!")
                st.session_state.stage = 'upload_test'
                st.experimental_rerun()

# 4️⃣ Test verisi yükle ve tahmin
if st.session_state.stage == 'upload_test':
    st.header("4️⃣ Test Verisi Yükleyin (.csv)")
    uploaded_test_file = st.file_uploader("Test CSV", type="csv")
    if uploaded_test_file is not None:
        df_test = load_csv(uploaded_test_file)
        if df_test is not None:
            st.session_state.test_df = df_test
            st.success("Test verisi yüklendi!")
            st.dataframe(df_test.head())
            st.session_state.stage = 'show_shap'
            st.experimental_rerun()

# 5️⃣ SHAP Analizi ve Tahmin
if st.session_state.stage == 'show_shap':
    st.header("5️⃣ SHAP Analizi ve Tahmin")
    X_test = st.session_state.test_df.drop(columns=[st.session_state.target_column], errors='ignore')
    y_test = st.session_state.test_df[st.session_state.target_column] if st.session_state.target_column in st.session_state.test_df else None
    model = st.session_state.model
    explainer, shap_values = plot_shap_summary(model, X_test)
    plot_shap_force(explainer, shap_values, X_test)

    if st.button("Yeni Analiz Başlat"):
        st.session_state.clear()
        st.experimental_rerun()
