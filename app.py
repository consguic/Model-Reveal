import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.datasets import load_iris, make_classification, make_blobs
from sklearn.model_selection import train_test_split

# Streamlit'in varsayılan temasına uyum sağlamak için özel CSS'i minimalist tutuyoruz.
st.set_page_config(
    page_title="Model Reveal - XAI Dashboard", 
    layout="wide"
)

# --- Sabitler ve Yardımcı Fonksiyonlar ---

# Model ve verileri önbelleğe al
@st.cache_resource
def load_models_and_data(model_type):
    """Belirlenen model türüne göre veri setini ve modeli yükler/eğitir."""
    
    if model_type == "Supervised (Sınıflandırma - Random Forest)":
        # Örnek Supervised (Gözetimli) Veri Seti (İris yerine daha genel bir sınıflandırma)
        X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_redundant=0, 
                                   n_classes=2, random_state=42)
        feature_names = [f'Özellik_{i+1}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Modeli Eğit
        X_train, _, y_train, _ = train_test_split(X_df, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        return model, X_df, {0: 'Sınıf 0 (Normal)', 1: 'Sınıf 1 (Yüksek Risk)'}

    else: # Unsupervised (Anomali Tespiti - Isolation Forest)
        # Örnek Unsupervised (Gözetimsiz) Veri Seti (Kümeleme ve Anomali)
        X, _ = make_blobs(n_samples=500, centers=1, cluster_std=1.0, random_state=42)
        # Yapay aykırı değerler ekle
        outliers = np.random.uniform(low=-10, high=10, size=(10, 2))
        X = np.concatenate([X, outliers], axis=0)
        
        feature_names = ['Değer_X', 'Değer_Y']
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Modeli Eğit
        # contamination='auto' veya bir float değeri (aykırı değer oranı) ayarlanabilir.
        model = IsolationForest(random_state=42)
        model.fit(X_df)
        
        return model, X_df, {-1: 'Anomali (Aykırı Değer)', 1: 'Normal'}

def plot_shap_waterfall(explainer, shap_values, input_df, class_index, title):
    """Tek bir örnek için SHAP Waterfall grafiği çizer."""
    
    # SHAP Explanation objesi oluştur
    # TreeExplainer'dan gelen shap_values çok boyutlu olabilir (sınıflandırma için)
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[class_index][0]
        base_value = explainer.expected_value[class_index]
    else:
        shap_values_to_plot = shap_values[0]
        base_value = explainer.expected_value
        
    # SHAP Explanation objesi oluşturulurken veri ve özellik isimleri verilir
    explanation = shap.Explanation(
        values=shap_values_to_plot,
        base_values=base_value,
        data=input_df.iloc[0].values,
        feature_names=input_df.columns.tolist()
    )
    
    # Waterfall plot çizimi
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    ax.set_title(title, fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

# --- UI Yapısı ---

st.title("💡 Model Reveal: XAI Karar Mekanizması Analizi")
st.markdown("İnteraktif bir Streamlit uygulamasıyla ML modelinizin neden o çıktıyı verdiğini keşfedin.")

st.sidebar.header("Uygulama Ayarları")
selected_model_type = st.sidebar.selectbox(
    "1. Model Türünü Seçin",
    ["Supervised (Sınıflandırma - Random Forest)", "Unsupervised (Anomali Tespiti - Isolation Forest)"]
)

model, X_df, class_map = load_models_and_data(selected_model_type)
st.sidebar.success(f"Model ({selected_model_type.split(' - ')[1]}) ve Veri Yüklendi.")

st.sidebar.markdown("---")
st.sidebar.header("2. Yeni Tahmin için Veri Girin")

# Kullanıcı Girdileri
input_data = {}
for col in X_df.columns:
    min_val = float(X_df[col].min())
    max_val = float(X_df[col].max())
    # Ortalama değeri varsayılan olarak ayarla
    default_val = float(X_df[col].mean())
    input_data[col] = st.sidebar.slider(
        col, 
        min_value=min_val, 
        max_value=max_val, 
        value=default_val, 
        step=(max_val - min_val)/100 # Daha hassas adım
    )

input_df = pd.DataFrame([input_data])

# --- Tahmin Butonu ve Sonuç Alanı ---
if st.sidebar.button('Modeli Açığa Çıkar (Reveal Tahmini)', type="primary"):
    
    # 1. Tahmin Yap
    if selected_model_type == "Supervised (Sınıflandırma - Random Forest)":
        # Sınıflandırma Tahmini
        prediction_value = model.predict(input_df)[0]
        prediction_label = class_map[prediction_value]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)[0]
            confidence = max(proba) * 100
            
        # 2. Tahmin ve Nedenleri Yan Yana Göster
        col_tahmin, col_shap = st.columns([1, 2])
        
        with col_tahmin:
            st.subheader("🎯 Tahmin Sonucu")
            st.success(f"Sınıf: **{prediction_label}**")
            st.metric(label="Güvenirlik", value=f"{confidence:.2f}%")
            
            st.markdown("---")
            st.subheader("Girdi Değerleri")
            st.dataframe(input_df.T, use_container_width=True)


        with col_shap:
            st.subheader("📊 SHAP Açıklaması (Neden?)")
            
            # SHAP Explainer oluştur
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            # SHAP'ı sadece tahmin edilen sınıf için görselleştir
            plot_shap_waterfall(
                explainer, 
                shap_values, 
                input_df, 
                prediction_value, 
                f"'{prediction_label}' Tahminine Katkılar"
            )
            st.caption("Grafik, tahminin temel ortalama çıktıdan nasıl saptığını ve hangi özelliklerin bu sapmaya ne kadar katkı sağladığını gösterir.")

    else: # Unsupervised (Anomali Tespiti - Isolation Forest)
        # Anomali Tespiti Tahmini
        # Isolation Forest'ta -1: Anomali, 1: Normal'dir.
        prediction_value = model.predict(input_df)[0]
        prediction_label = class_map[prediction_value]
        
        # Anomali skorunu al (daha düşük skor = daha yüksek anomali)
        anomaly_score = model.decision_function(input_df)[0]
        
        col_tahmin, col_shap = st.columns([1, 2])

        with col_tahmin:
            st.subheader("🎯 Tespit Sonucu")
            if prediction_value == -1:
                st.error(f"Tespit: **{prediction_label}**")
            else:
                st.info(f"Tespit: **{prediction_label}**")
            
            st.metric(label="Anomali Skoru", value=f"{anomaly_score:.4f}", delta=f"{-anomaly_score:.4f} (Normalden Uzaklık)", delta_color="inverse")
            
            st.markdown("---")
            st.subheader("Girdi Değerleri")
            st.dataframe(input_df.T, use_container_width=True)

        with col_shap:
            st.subheader("📊 SHAP Açıklaması (Anomaliye Neden?)")
            
            # SHAP IsolationForest için biraz farklı çalışır (skor bazlı)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            plot_shap_waterfall(
                explainer, 
                shap_values, 
                input_df, 
                0, # IsolationForest SHAP değeri tek boyutludur, 0. indisi kullan
                f"'{prediction_label}' Tespitine Özellik Katkıları"
            )
            st.caption("Grafik, modelin neden bu örneği **Normal** ya da **Anomali** olarak skorladığını gösterir. Pozitif katkı normalliği, negatif katkı anomaliyi destekler (Isolation Forest skorlamasına bağlı olarak değişebilir).")

else:
    st.info("Lütfen sol panelden (Sidebar) model türünü seçin ve parametreleri ayarlayıp 'Modeli Açığa Çıkar' butonuna tıklayın.")

st.markdown("---")
st.markdown("💡 Model Reveal – XAI Dashboard | SHAP & Streamlit Entegrasyonu")
