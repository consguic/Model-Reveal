# app.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# 🌿 Sayfa Ayarları ve Stil
# -------------------------------
st.set_page_config(page_title="GlassBox AI Dashboard", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
    font-family: 'Arial', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("🌿 GlassBox – AI Explainability Dashboard")
st.write("Şeffaf yapay zekâ deneyimi: Modelinizin kararlarını keşfedin!")

# -------------------------------
# 🌟 Model Seçimi / Yükleme
# -------------------------------
st.sidebar.header("Model Seçimi")

mode = st.sidebar.radio("Mod Seçin:", ("Demo Model (Iris)", "Kendi Modelini Yükle"))

if mode == "Demo Model (Iris)":
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    model = RandomForestClassifier(random_state=42).fit(X, y)
    st.sidebar.success("Demo modeli yüklendi!")
else:
    uploaded_file = st.sidebar.file_uploader("Bir model (.pkl) dosyası yükle", type=["pkl"])
    if uploaded_file is not None:
        model = joblib.load(uploaded_file)
        # Özellik isimleri otomatik algılama
        if hasattr(model, 'feature_names_in_'):
            X_columns = model.feature_names_in_
        else:
            st.warning("Modelin feature isimleri yok. Demo Iris kullanılıyor.")
            iris = load_iris()
            X = pd.DataFrame(iris.data, columns=iris.feature_names)
            y = iris.target
            model = RandomForestClassifier(random_state=42).fit(X, y)
            X_columns = X.columns
        st.sidebar.success("Model başarıyla yüklendi!")

# -------------------------------
# 📝 Kullanıcı Girdisi
# -------------------------------
st.sidebar.header("Yeni Tahmin için Veri Girin")

input_data = {}
if mode == "Demo Model (Iris)" or uploaded_file is not None:
    # Kolon isimleri
    columns = X.columns if mode=="Demo Model (Iris)" else X_columns
    for col in columns:
        min_val = float(X[col].min()) if mode=="Demo Model (Iris)" else 0.0
        max_val = float(X[col].max()) if mode=="Demo Model (Iris)" else 10.0
        step = (max_val - min_val)/100
        input_data[col] = st.sidebar.number_input(col, min_value=min_val, max_value=max_val, value=(min_val+max_val)/2, step=step)
    input_df = pd.DataFrame([input_data])

    # -------------------------------
    # 🔮 Tahmin
    # -------------------------------
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔮 Tahmin Sonucu")
        if mode=="Demo Model (Iris)":
            st.metric(label="Tahmin", value=iris.target_names[prediction])
            if proba is not None:
                st.metric(label="Confidence", value=f"{max(proba)*100:.2f}%")
        else:
            st.metric(label="Tahmin", value=str(prediction))
            if proba is not None:
                st.metric(label="Confidence", value=f"{max(proba)*100:.2f}%")

    # -------------------------------
    # 📊 SHAP Açıklaması
    # -------------------------------
    with col2:
        st.subheader("📊 Özellik Etkileri (SHAP)")

        # SHAP Explainer
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X) if mode=="Demo Model (Iris)" else explainer.shap_values(input_df)
            # Summary plot
            plt.figure(figsize=(6,4))
            if mode=="Demo Model (Iris)":
                shap.summary_plot(shap_values, X, show=False)
            else:
                shap.summary_plot(shap_values, input_df, show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.warning(f"SHAP açıklaması oluşturulamadı: {e}")

else:
    st.info("Lütfen sol taraftan model seçin veya yükleyin.")

# -------------------------------
# 🔹 Footer / Ek Bilgi
# -------------------------------
st.markdown("---")
st.markdown("💡 GlassBox – AI Explainability Dashboard | Made with ❤️ by Fatma Kızılkaya")
