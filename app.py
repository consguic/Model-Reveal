import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="ModelReveal", layout="wide")
st.title("🌿 Model Reveal – AI Explainability Dashboard")

# Modeli hazırla
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
model = RandomForestClassifier().fit(X, y)

# Kullanıcıdan girdi al
st.sidebar.header("Yeni tahmin için veri gir:")
inputs = []
for feature in X.columns:
    val = st.sidebar.slider(feature, float(X[feature].min()), float(X[feature].max()))
    inputs.append(val)

# Tahmin
input_df = pd.DataFrame([inputs], columns=X.columns)
prediction = model.predict(input_df)[0]
st.write(f"### 🔮 Model Tahmini: **{iris.target_names[prediction]}**")

# SHAP açıklaması
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
st.subheader("📊 Model Özellik Etkileri (SHAP)")


# SHAP summary plot
shap.summary_plot(shap_values, X)

st.pyplot(plt.gcf())  # plt.gcf() = current figure
plt.clf()  # Sonraki plot için temizle
st.pyplot(bbox_inches='tight')
