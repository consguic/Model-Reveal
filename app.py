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
for key in ["stage","model","X_train","y_train","le","feature_names",
            "model_type","train_df","test_df","target_column"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "temp_target" not in st.session_state:
    st.session_state.temp_target = None

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
    except:
        st.error("CSV yüklenirken hata oluştu")
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
    if model_class in [RandomForestClassifier, XGBClassifier]:
        model = model_class(random_state=42, n_estimators=100)
    else:
        model = model_class(random_state=42)
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
    shap.force_plot(
        explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_val,
        X.iloc[0,:],
        matplotlib=True
    )
    st.pyplot(plt.gcf())
    plt.close()

# --- Callback fonksiyonları ---
def select_target_callback():
    st.session_state.target_column = st.session_state.temp_target
    st.session_state.stage = 'select_model'

def train_model_callback(model_name):
    X, y = preprocess(st.session_state.train_df, st.session_state.target_column)
    st.session_state.X_train = X
    st.session_state.y_train = y
    st.session_state.model = train_model(X, y, MODEL_OPTIONS[model_name])
    st.session_state.model_type = model_name
    st.session_state.stage = 'upload_test'

def reset_app():
    keys = ["stage","model","X_train","y_train","le","feature_names",
            "model_type","train_df","test_df","target_column","temp_target"]
    for key in keys:
        st.session_state[key] = None

# --- Aşamalar ---
# 1️⃣ Eğitim CSV
if st.session_state.stage in [None,'upload_train']:
    st.header("1️⃣ Eğitim Verisi Yükleyin (.csv)")
    uploaded_train_file = st.file_uploader("Eğitim CSV", type="csv")
    if uploaded_train_file is not None:
        df = load_csv(uploaded_train_file)
        if df is not None:
            st.session_state.train_df = df
            st.session_state.temp_target = df.columns[0]
            st.success("Eğitim verisi yüklendi!")
            st.dataframe(df.head())
            st.session_state.stage = 'select_target'

# 2️⃣ Target seçimi
if st.session_state.stage == 'select_target':
    st.header("2️⃣ Target Sütunu Seçin")
    st.selectbox("Target Sütunu", st.session_state.train_df.columns, key="temp_target")
    st.button("Onayla ve Model Seç", on_click=select_target_callback)

# 3️⃣ Model seçimi
if st.session_state.stage == 'select_model':
    st.header("3️⃣ Model Seçin ve Eğitin")
    col1, col2, col3 = st.columns(3)
    for idx, (name, cls) in enumerate(MODEL_OPTIONS.items()):
        with [col1, col2, col3][idx]:
            st.button(f"{name} Eğit", on_click=train_model_callback, args=(name,))

# 4️⃣ Test verisi
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

# 5️⃣ SHAP Analizi
if st.session_state.stage == 'show_shap':
    st.header("5️⃣ SHAP Analizi ve Tahmin")
    X_test = st.session_state.test_df.drop(columns=[st.session_state.target_column], errors='ignore')
    model = st.session_state.model
    explainer, shap_values = plot_shap_summary(model, X_test)
    plot_shap_force(explainer, shap_values, X_test)
    st.button("Yeni Analiz Başlat", on_click=reset_app)
