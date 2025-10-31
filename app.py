import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Model Reveal", layout="wide")

# --- Güvenli rerun fonksiyonu ---
def safe_rerun():
    try:
        st.experimental_rerun()
    except:
        pass

# --- SHAP grafiği ---
def plot_shap_waterfall(explainer, shap_values, input_df, prediction_label):
    base_value = getattr(explainer, "expected_value", 0)
    shap_values_plot = shap_values.values[0] if hasattr(shap_values, "values") else shap_values[0]
    data = input_df.iloc[0].values
    
    exp = shap.Explanation(
        values=shap_values_plot,
        base_values=base_value,
        data=data,
        feature_names=input_df.columns
    )
    shap.plots.waterfall(exp, show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

# --- Basit state ---
if 'stage' not in st.session_state:
    st.session_state.stage = 'upload_train'

st.title("💡 Model Reveal: Explainable Anomaly Dashboard")

if st.session_state.stage == 'upload_train':
    uploaded = st.file_uploader("Eğitim verisini yükle (.csv)", type='csv')
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        st.success("Veri yüklendi!")
        st.dataframe(df.head())
        if st.button("Isolation Forest ile eğit"):
            X = pd.get_dummies(df, drop_first=True)
            model = IsolationForest(random_state=42)
            model.fit(X)
            st.session_state.model = model
            st.session_state.X_train = X
            st.session_state.stage = 'upload_test'
            safe_rerun()

elif st.session_state.stage == 'upload_test':
    st.header("Test verisini yükleyin (.csv)")
    uploaded_test = st.file_uploader("Tek örnek içermeli", type='csv')
    if uploaded_test:
        test = pd.read_csv(uploaded_test)
        test_X = pd.get_dummies(test, drop_first=True).reindex(columns=st.session_state.X_train.columns, fill_value=0)
        model = st.session_state.model
        pred = model.predict(test_X)[0]
        score = model.decision_function(test_X)[0]
        label = "Anomali" if pred == -1 else "Normal"
        st.subheader(f"Tespit: **{label}** (skor: {score:.3f})")
        
        st.subheader("📊 SHAP Açıklaması")
        explainer = shap.Explainer(model, st.session_state.X_train)
        shap_values = explainer(test_X)
        plot_shap_waterfall(explainer, shap_values, test_X, label)
