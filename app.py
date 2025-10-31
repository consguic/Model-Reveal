import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib # Modelleri kaydetmek/yüklemek için kullanılabilir (şimdilik demo amaçlı eğitiyoruz)

# --- Sabitler ve Model Tanımları ---

# Desteklenen Modeller
MODEL_CHOICES = {
    "Supervised (Sınıflandırma)": {
        "Random Forest Sınıflandırıcı": RandomForestClassifier,
        "Lojistik Regresyon": LogisticRegression
    },
    "Unsupervised (Anomali Tespiti)": {
        "Isolation Forest": IsolationForest,
        "One-Class SVM": OneClassSVM # Daha stabil bir model için Kernel Explainer gerektirebilir
    }
}

# --- Session State Başlatma ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'target_encoder' not in st.session_state:
    st.session_state.target_encoder = None
if 'class_map' not in st.session_state:
    st.session_state.class_map = None


# --- Yardımcı Fonksiyonlar ---

def plot_shap_waterfall(explainer, shap_values, input_df, class_index, title):
    """Tek bir örnek için SHAP Waterfall grafiği çizer."""
    
    # SHAP Explanation objesi oluşturma (TreeExplainer, KernelExplainer vb. için uyumlu)
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[class_index][0]
        base_value = explainer.expected_value[class_index]
    else:
        # Unsupervised/Regresyon durumları için tek boyutlu değerler
        shap_values_to_plot = shap_values[0]
        base_value = explainer.expected_value
        
    explanation = shap.Explanation(
        values=shap_values_to_plot,
        base_values=base_value,
        data=input_df.iloc[0].values,
        feature_names=input_df.columns.tolist()
    )
    
    # Waterfall plot çizimi
    # Matplotlib figürü oluşturulur ve Streamlit'e aktarılır.
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    ax.set_title(title, fontsize=14)
    st.pyplot(fig)
    plt.close(fig)


# --- UI Yapısı ---
st.set_page_config(
    page_title="Model Reveal - XAI Dashboard", 
    layout="wide"
)

st.title("💡 Model Reveal: XAI Karar Mekanizması Analizi")
st.markdown("Veri setinizi yükleyin, modelinizi eğitin ve tahminlerinin ardındaki nedenleri SHAP ile keşfedin.")

# --- SIDEBAR: Uygulama Ayarları ve Veri Yükleme ---

st.sidebar.header("1. Veri Setini Yükleyin (.csv)")
uploaded_file = st.sidebar.file_uploader("Eğitim Verisi Yükle", type=["csv"])

if uploaded_file is not None and not st.session_state.data_loaded:
    try:
        data = pd.read_csv(uploaded_file)
        st.session_state.original_data = data
        st.session_state.data_loaded = True
        st.sidebar.success("Veri başarıyla yüklendi!")
    except Exception as e:
        st.sidebar.error(f"Dosya yüklenirken hata oluştu: {e}")

if st.session_state.data_loaded:
    data = st.session_state.original_data
    st.sidebar.subheader("2. Model Seçimi")
    
    # Model Türü Seçimi
    model_category = st.sidebar.selectbox(
        "Model Kategorisi",
        list(MODEL_CHOICES.keys())
    )
    
    # Spesifik Model Seçimi
    selected_model_name = st.sidebar.selectbox(
        "Spesifik Model",
        list(MODEL_CHOICES[model_category].keys())
    )
    
    # Etiket (Target) Sütun Seçimi
    if model_category == "Supervised (Sınıflandırma)":
        target_column = st.sidebar.selectbox(
            "Etiket (Target) Sütunu Seçin",
            data.columns
        )
        st.sidebar.info("Supervised model seçtiğiniz için, veri setinizde Etiket Sütununun (Target) mevcut olması ve doğru seçilmesi ZORUNLUDUR.")
    else:
        target_column = None
        st.sidebar.info("Unsupervised model seçtiniz. Veri setiniz etiket (target) içermemelidir.")
        
    # --- Model Eğitme Butonu ---
    if st.sidebar.button("3. Modeli Eğit ve Hazırla", type="primary"):
        # Veri Hazırlığı
        if target_column:
            # Supervised
            try:
                # Sayısal olmayan sütunları hariç tut
                X = data.drop(columns=[target_column]).select_dtypes(include=np.number)
                y = data[target_column]
                
                # Etiketi sayısal hale getir (Label Encoding)
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                st.session_state.target_encoder = le
                st.session_state.class_map = {i: label for i, label in enumerate(le.classes_)}

                # Özellikleri ölçeklendir
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                st.session_state.scaler = scaler
                st.session_state.feature_names = X.columns.tolist()
                
                # Modeli Eğit
                ModelClass = MODEL_CHOICES[model_category][selected_model_name]
                model_instance = ModelClass(random_state=42) if 'random_state' in ModelClass().get_params() else ModelClass()
                model_instance.fit(X_scaled, y_encoded)
                
                st.session_state.trained_model = model_instance
                st.session_state.model_trained = True
                st.balloons()
                st.sidebar.success(f"{selected_model_name} modeli başarıyla eğitildi ve hazır.")
                
            except Exception as e:
                st.sidebar.error(f"Supervised model eğitimi sırasında hata: {e}")
                st.session_state.model_trained = False
        
        else:
            # Unsupervised
            try:
                # Sayısal olmayan sütunları hariç tut ve etiket yok say
                X = data.select_dtypes(include=np.number)
                st.session_state.feature_names = X.columns.tolist()

                # Özellikleri ölçeklendir
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                st.session_state.scaler = scaler
                
                # Modeli Eğit
                ModelClass = MODEL_CHOICES[model_category][selected_model_name]
                model_instance = ModelClass(random_state=42) if 'random_state' in ModelClass().get_params() else ModelClass()
                model_instance.fit(X_scaled)
                
                st.session_state.trained_model = model_instance
                st.session_state.model_trained = True
                st.balloons()
                st.sidebar.success(f"{selected_model_name} modeli başarıyla eğitildi ve hazır.")
            except Exception as e:
                st.sidebar.error(f"Unsupervised model eğitimi sırasında hata: {e}")
                st.session_state.model_trained = False

# --- Ana Gövde: Eğitim ve Test Aşamaları ---

if not st.session_state.data_loaded:
    st.info("Lütfen sol menüden (sidebar) eğitim veri setinizi yükleyerek başlayın.")
    
elif not st.session_state.model_trained:
    st.warning("Veri yüklendi. Lütfen bir model seçin, Etiket Sütununu (Supervised için) belirleyin ve Modeli Eğit butonuna tıklayın.")

elif st.session_state.model_trained:
    st.success(f"Eğitim Başarılı! **{selected_model_name}** modeli teste hazır. ")

    st.header("4. Test Verisi Yükle ve Açıklamayı Gör")
    test_file = st.file_uploader("Test Verisi Yükle (Tek bir örnek içeren .csv önerilir)", type=["csv"])
    
    if test_file is not None:
        try:
            test_data = pd.read_csv(test_file)
            
            # Sadece eğitimde kullanılan sütunları al
            test_data_X = test_data[st.session_state.feature_names]
            
            # Veriyi ölçeklendir
            test_data_scaled = pd.DataFrame(
                st.session_state.scaler.transform(test_data_X), 
                columns=st.session_state.feature_names
            )
            
            # --- Tahmin Yap ve SHAP'ı Hesapla ---
            model = st.session_state.trained_model
            
            # SHAP sadece ilk satırı açıklar (tek örnek testi için)
            sample_to_explain = test_data_scaled.iloc[[0]]
            original_input = test_data_X.iloc[[0]]
            
            # 1. Tahmin Yap
            prediction_value = model.predict(sample_to_explain)[0]
            
            # 2. SHAP Açıklaması
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(sample_to_explain)
            except Exception:
                # TreeExplainer desteklemeyen modeller için (örn. Lojistik Regresyon, OneClassSVM) Kernel Explainer dene.
                # Kernel Explainer yavaştır, bu yüzden sadece gerektiğinde kullanılır.
                st.warning("TreeExplainer desteklenmiyor. Kernel Explainer kullanılıyor. Bu işlem biraz zaman alabilir.")
                # SHAP için bir 'arka plan' verisi (eğitim setinden örnek) gereklidir.
                background = st.session_state.original_data.sample(100, random_state=42).select_dtypes(include=np.number)
                background_scaled = st.session_state.scaler.transform(background)
                explainer = shap.KernelExplainer(model.predict, background_scaled)
                shap_values = explainer.shap_values(sample_to_explain)
            
            # --- Sonuçları Göster ---
            
            col_tahmin, col_shap = st.columns([1, 2])
            
            with col_tahmin:
                st.subheader("🎯 Tahmin Sonucu")
                if isinstance(model, (RandomForestClassifier, LogisticRegression)):
                    # Supervised Sınıflandırma
                    prediction_label = st.session_state.class_map[prediction_value]
                    st.success(f"Sınıf: **{prediction_label}**")
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(sample_to_explain)[0]
                        confidence = max(proba) * 100
                        st.metric(label="Güvenirlik", value=f"{confidence:.2f}%")

                    plot_shap_waterfall(
                        explainer, 
                        shap_values, 
                        original_input, 
                        prediction_value, 
                        f"'{prediction_label}' Tahminine Katkılar"
                    )

                elif isinstance(model, (IsolationForest, OneClassSVM)):
                    # Unsupervised Anomali Tespiti
                    prediction_label = {-1: 'Anomali (Aykırı Değer)', 1: 'Normal'}[prediction_value]
                    anomaly_score = model.decision_function(sample_to_explain)[0]
                    
                    if prediction_value == -1:
                        st.error(f"Tespit: **{prediction_label}**")
                    else:
                        st.info(f"Tespit: **{prediction_label}**")
                        
                    st.metric(label="Anomali Skoru", value=f"{anomaly_score:.4f}", delta=f"{-anomaly_score:.4f} (Normalden Uzaklık)", delta_color="inverse")
                    
                    # Isolation Forest için SHAP görselleştirmesi
                    plot_shap_waterfall(
                        explainer, 
                        shap_values, 
                        original_input, 
                        0, 
                        f"'{prediction_label}' Tespitine Özellik Katkıları"
                    )
            
            with col_shap:
                st.subheader("📊 SHAP Açıklaması (Neden?)")
                st.markdown("**Bu Örnek İçin Girdi Değerleri:**")
                st.dataframe(original_input.T, use_container_width=True)
                st.markdown("---")
                st.markdown("Grafik, modelin temel ortalama çıktıdan (Base Value) nasıl saptığını ve hangi özelliklerin bu sapmaya ne kadar katkı sağladığını gösterir.")

        except Exception as e:
            st.error(f"Test verisi veya açıklama oluşturulurken hata oluştu: {e}")
            st.warning("Lütfen test dosyanızın, eğitim dosyanızla aynı sütunlara (etiket hariç) sahip olduğundan emin olun.")

st.markdown("---")
st.markdown("💡 Model Reveal – XAI Dashboard | SHAP & Streamlit Entegrasyonu")
