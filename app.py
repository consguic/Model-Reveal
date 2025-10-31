import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Streamlit session state'i başlat
if 'stage' not in st.session_state:
    st.session_state.stage = 'upload_train'
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'le' not in st.session_state: # Label Encoder (Target için)
    st.session_state.le = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None


# --- Sabitler ve Yardımcı Fonksiyonlar ---

MODEL_OPTIONS = {
    "Supervised (Sınıflandırma)": {
        "Random Forest Sınıflandırıcı": RandomForestClassifier,
        "Gradient Boosting Sınıflandırıcı": None
    },
    "Unsupervised (Anomali Tespiti)": {
        "Isolation Forest": IsolationForest,
        "One-Class SVM": None
    }
}

@st.cache_data
def handle_uploaded_data(uploaded_file, skip_target=False):
    """Yüklenen CSV dosyasını okur ve temel veri temizliğini yapar."""
    try:
        # Yorum satırlarını (# ile başlayanları) atlayarak oku
        df = pd.read_csv(uploaded_file, comment='#')
        
        # Sadece sayısal ve kategorik veri türlerini tut, NaN olanları ortalama ile doldur (basit yaklaşım)
        for col in df.columns:
            if df[col].dtype == 'object':
                # Eğer target sütunu değilse ve skip_target True ise, bu sütunu atla
                if skip_target and col == st.session_state.target_column:
                    continue
                # Kategorik sütunları One-Hot Encoding için bırak
                continue
            
            # Sayısal sütunlardaki eksik değerleri (NaN) ortalama ile doldur
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        return None

def preprocess_data_and_train(df, model_class, target_column=None):
    """Veriyi ön işler ve modeli eğitir."""
    
    # Kategori: One-Hot Encoding
    X = pd.get_dummies(df.drop(columns=[target_column]) if target_column else df, drop_first=True)
    
    if target_column: # Supervised (Gözetimli)
        y = df[target_column]
        
        # Etiket (Target) sütununu sayısal hale getir (Label Encoding)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        st.session_state.le = le
        st.session_state.feature_names = X.columns.tolist()

        # Modeli Eğit
        X_train, _, y_train, _ = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        model = model_class(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        
        # Eğitim verilerini session state'e kaydet (SHAP için gerekli)
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        
        return model, X_train.columns.tolist()
    
    else: # Unsupervised (Gözetimsiz)
        
        # Modeli Eğit (tüm veri ile)
        model = model_class(random_state=42)
        model.fit(X)
        
        st.session_state.X_train = X # SHAP için tüm X verisini kullan
        st.session_state.feature_names = X.columns.tolist()
        
        return model, X.columns.tolist()


def plot_shap_waterfall(explainer, shap_values, input_df, class_index, title, prediction_label):
    """Tek bir örnek için SHAP Waterfall grafiği çizer."""
    
    # SHAP Explanation objesi için değerleri ve beklenen değeri al
    if isinstance(shap_values, list):
        # Sınıflandırma modelleri için: Sadece tahmin edilen sınıfa ait SHAP değerlerini kullan
        shap_values_to_plot = shap_values[class_index][0]
        base_value = explainer.expected_value[class_index]
    else:
        # Anomali modelleri için: Tek bir çıktı vektörü vardır
        shap_values_to_plot = shap_values[0]
        base_value = explainer.expected_value
        
    explanation = shap.Explanation(
        values=shap_values_to_plot,
        base_values=base_value,
        data=input_df.iloc[0].values,
        feature_names=input_df.columns.tolist()
    )
    
    # Waterfall plot çizimi
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    ax.set_title(f"'{prediction_label}' Tahminine Katkılar", fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

# --- UI Yapısı ---

st.title("💡 Model Reveal: XAI Karar Mekanizması Analizi")
st.markdown("İnteraktif bir Streamlit uygulamasıyla ML modelinizin neden o çıktıyı verdiğini keşfedin.")

# --- Sol Panel (Ayarlar ve Aşamalar) ---

st.sidebar.header("Uygulama Akışı")
st.sidebar.markdown(f"**Güncel Aşama:** `{st.session_state.stage}`")
st.sidebar.markdown("---")


# ===============================================
# AŞAMA 1: EĞİTİM VERİSİ YÜKLE
# ===============================================
if st.session_state.stage == 'upload_train':
    st.header("1. Eğitim Verisini Yükleyin (.csv)")
    uploaded_train_file = st.file_uploader(
        "Lütfen modelinizi eğitmek için .csv dosyanızı yükleyin (İlk 15 satırlık Kredi Riski verisini kullanabilirsiniz).", 
        type="csv"
    )
    
    if uploaded_train_file is not None:
        train_df = handle_uploaded_data(uploaded_train_file)
        if train_df is not None:
            st.session_state.train_df = train_df
            st.success("Eğitim verisi başarıyla yüklendi!")
            st.dataframe(train_df.head(), use_container_width=True)
            st.session_state.stage = 'select_model'
            st.rerun() # Yeni aşamaya geç

# ===============================================
# AŞAMA 2: MODEL SEÇİMİ VE EĞİTİM
# ===============================================
if st.session_state.stage == 'select_model':
    st.header("2. Model Seçimi ve Eğitimi")
    
    # Model Türü Seçimi
    model_category = st.selectbox(
        "Model Kategorisi",
        list(MODEL_OPTIONS.keys())
    )
    
    # Model Sınıfı Seçimi
    model_name = st.selectbox(
        "Model Sınıfı",
        list(MODEL_OPTIONS[model_category].keys()),
        disabled=(MODEL_OPTIONS[model_category].get(list(MODEL_OPTIONS[model_category].keys())[0]) is None) # Eğer model None ise pasif yap
    )
    
    st.session_state.model_type = model_category
    st.session_state.model_class = MODEL_OPTIONS[model_category][model_name]
    
    if model_category == "Supervised (Sınıflandırma)":
        
        # Etiket (Target) Sütunu Seçimi
        target_column = st.selectbox(
            "Etiket (Target) Sütununu Seçin",
            st.session_state.train_df.columns
        )
        st.session_state.target_column = target_column

        # Etiket kontrolü
        if st.session_state.train_df[target_column].dtype not in ['int64', 'float64']:
            st.info(f"Seçilen '{target_column}' sütunu metin içeriyor. Model eğitimi öncesinde Label Encoding uygulanacaktır.")
        
        # Özellikler kontrolü (Etiket hariç hepsi sayısal/kategorik olmalı)
        feature_cols = st.session_state.train_df.drop(columns=[target_column]).columns
        
        st.info("Supervised model seçtiniz. Veri setinizde bir etiket (target) sütunu olmalıdır. Model eğitimi, kategori dönüşümü (One-Hot Encoding) yapılarak gerçekleştirilecektir.")
        
    else:
        st.info("Unsupervised model seçtiniz. Veri setinizde etiket (target) sütunu gerekli değildir. Model (Isolation Forest), verideki aykırı noktaları tespit etmek için eğitilecektir.")

    if st.button(f"'{model_name}' Modelini Eğit", type="primary"):
        with st.spinner("Model eğitiliyor ve veri önişleniyor..."):
            try:
                # Veri önişleme ve eğitim
                st.session_state.model, feature_names = preprocess_data_and_train(
                    st.session_state.train_df, 
                    st.session_state.model_class, 
                    st.session_state.target_column if model_category == "Supervised (Sınıflandırma)" else None
                )
                st.success(f"{model_name} başarıyla eğitildi!")
                st.session_state.feature_names = feature_names
                st.session_state.stage = 'upload_test'
                st.rerun()
            except Exception as e:
                st.error(f"Model eğitimi sırasında hata oluştu: {e}. Lütfen veri tiplerini kontrol edin.")

# ===============================================
# AŞAMA 3: TEST VERİSİ YÜKLEME VE TAHMİN
# ===============================================
if st.session_state.stage == 'upload_test':
    st.header("3. Test Verisini Yükleyin (.csv)")
    st.info(f"Eğitilen model: **{st.session_state.model_type}** | Yükleyeceğiniz veri, eğitim verisindeki sütunlarla ({len(st.session_state.feature_names)} özellik) uyumlu olmalıdır.")
    
    uploaded_test_file = st.file_uploader(
        "Tahmin ve XAI analizi için tek bir örnek içeren test .csv dosyanızı yükleyin (Risk sütunu olmamalıdır).", 
        type="csv"
    )
    
    if uploaded_test_file is not None:
        test_df = handle_uploaded_data(uploaded_test_file, skip_target=True)
        if test_df is not None:
            
            # Test verisine eğitim verisine uygulanan dönüşümleri uygula (One-Hot Encoding)
            test_X_raw = test_df.copy()
            test_X_processed = pd.get_dummies(test_X_raw, drop_first=True).reindex(columns=st.session_state.feature_names, fill_value=0)
            
            if test_X_processed.shape[0] != 1:
                st.warning("Test verisi sadece tek bir örnek (tek bir satır) içermelidir.")
            else:
                st.session_state.test_X = test_X_processed
                st.session_state.test_X_raw = test_X_raw # SHAP'ta daha anlaşılır isimler için
                st.session_state.stage = 'reveal_result'
                st.rerun()

# ===============================================
# AŞAMA 4: SONUÇ VE XAI ANALİZİ
# ===============================================
if st.session_state.stage == 'reveal_result':
    st.header("4. Model Reveal: Sonuç ve Nedenleri")
    
    test_X_processed = st.session_state.test_X
    test_X_raw = st.session_state.test_X_raw
    model = st.session_state.model
    
    # 1. Tahmin Yap
    if st.session_state.model_type == "Supervised (Sınıflandırma)":
        # Sınıflandırma Tahmini
        prediction_value = model.predict(test_X_processed)[0]
        prediction_label = st.session_state.le.inverse_transform([prediction_value])[0] # Orijinal etikete geri çevir
        
        proba = model.predict_proba(test_X_processed)[0]
        confidence = max(proba) * 100
        
        # Tahmin ve Nedenleri Yan Yana Göster
        col_tahmin, col_shap = st.columns([1, 2])
        
        with col_tahmin:
            st.subheader("🎯 Tahmin Sonucu")
            st.success(f"Sınıf: **{prediction_label}**")
            st.metric(label="Güvenirlik", value=f"{confidence:.2f}%")
            
            st.markdown("---")
            st.subheader("Girdi Değerleri")
            st.dataframe(test_X_raw.T, use_container_width=True)


        with col_shap:
            st.subheader("📊 SHAP Açıklaması (Neden?)")
            
            # SHAP Explainer oluştur ve hesapla
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(test_X_processed)
            
            # SHAP'ı sadece tahmin edilen sınıf için görselleştir
            plot_shap_waterfall(
                explainer, 
                shap_values, 
                test_X_processed, # OHE'li veriyi kullan
                prediction_value, 
                f"'{prediction_label}' Tahminine Katkılar",
                prediction_label
            )
            st.caption("Grafik, tahminin temel ortalama çıktıdan nasıl saptığını ve hangi özelliklerin (OHE'li) bu sapmaya ne kadar katkı sağladığını gösterir.")

    else: # Unsupervised (Anomali Tespiti)
        # Anomali Tespiti Tahmini
        prediction_value = model.predict(test_X_processed)[0] # -1 veya 1
        prediction_label = "Anomali (Aykırı Değer)" if prediction_value == -1 else "Normal"
        
        # Anomali skorunu al (daha düşük skor = daha yüksek anomali)
        anomaly_score = model.decision_function(test_X_processed)[0]
        
        col_tahmin, col_shap = st.columns([1, 2])

        with col_tahmin:
            st.subheader("🎯 Tespit Sonucu")
            if prediction_value == -1:
                st.error(f"Tespit: **{prediction_label}**")
            else:
                st.success(f"Tespit: **{prediction_label}**")
            st.metric(label="Anomali Skoru", value=f"{anomaly_score:.4f}", delta=f"{'Daha Düşük = Daha Anormal' if anomaly_score < 0 else 'Daha Yüksek = Daha Normal'}")
            
            st.markdown("---")
            st.subheader("Girdi Değerleri")
            st.dataframe(test_X_raw.T, use_container_width=True)


        with col_shap:
            st.subheader("📊 SHAP Açıklaması (Neden?)")

            # SHAP Explainer oluştur ve hesapla
            explainer = shap.TreeExplainer(model)
            # Isolation Forest genellikle sadece tek bir çıktı değeri döndürür.
            shap_values = explainer.shap_values(test_X_processed) 
            
            plot_shap_waterfall(
                explainer, 
                np.array([shap_values]), # SHAP'a tek bir değer gibi sunmak için
                test_X_processed, 
                0, # Index önemsiz
                f"'{prediction_label}' Kararına Katkılar",
                prediction_label
            )
            st.caption("Grafik, örneğin neden anormal (negatif sapma) veya normal (pozitif sapma) olarak sınıflandırıldığına dair özellik katkılarını gösterir.")

    if st.button('Yeni Analiz Başlat', type="secondary"):
        st.session_state.stage = 'upload_train'
        st.session_state.model = None
        st.session_state.test_X = None
        st.session_state.le = None
        st.rerun()
