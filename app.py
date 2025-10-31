import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Sayfa Yapılandırması ve Tema ---
# Streamlit varsayılan olarak açık temayı (beyaz arka plan) kullanır.
st.set_page_config(
    page_title="Model Reveal",
    layout="wide", # Geniş layout kullan
    initial_sidebar_state="collapsed" # Yan paneli başlangıçta kapalı tut
)

# --- Özel CSS Enjeksiyonu ---
# Sadece buton rengini, başlık boyutunu ve sosyal medya ikonlarını özelleştiriyoruz.
st.markdown(
    """
    <style>
    /* Büyük başlık stili */
    h1 {
        font-size: 5em !important; 
        color: #000000; /* Koyu metin */
        line-height: 1.0;
        margin-top: 0.5em;
        margin-bottom: 0.1em;
    }
    /* Küçük metin stili */
    p {
        font-size: 1.2em !important;
        color: #000000;
        opacity: 0.8;
    }
    /* Sarı Butonlar */
    div.stButton > button {
        background-color: #FFD700; /* Sarı renk */
        color: #000000; /* Siyah metin */
        font-weight: bold;
        border: none;
        padding: 1em 2em;
        border-radius: 0.5em;
        font-size: 1.1em;
        margin: 0.5em 0;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #FFC000; /* Hover rengi */
    }
    /* Sosyal Medya ikonları için */
    .social-icons a {
        color: #000000; /* Siyah ikonlar */
        font-size: 1.5em;
        margin-right: 1em;
        text-decoration: none;
    }
    .social-icons a:hover {
        color: #FFD700; /* Hoverda sarı olsun */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Başlık Alanı ---
st.markdown("<h2 style='color:#000000; opacity:0.7;'>MODEL REVEAL</h2>", unsafe_allow_html=True)

# --- Ana İçerik ---
col1, col2 = st.columns([2, 1]) # Sol taraf geniş, sağ taraf dar

with col1:
    st.markdown("<h1>Explain the output</h1>", unsafe_allow_html=True) # Özel CSS ile büyüklük
    st.write("This web app helps understand models behaviour")

    st.markdown("<br>", unsafe_allow_html=True) # Boşluk bırak

    # Butonlar
    st.button("SELECT THE MACHINE LEARNING MODEL")
    st.button("UPLOAD YOUR DATA")

    st.markdown("<br><br><br>", unsafe_allow_html=True) # Daha fazla boşluk

    # Sosyal Medya
    st.markdown("---") # Yatay çizgi
    st.markdown("<p>GET SOCIAL</p>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="social-icons">
            <a href="https://twitter.com" target="_blank">🐦</a>
            <a href="https://linkedin.com" target="_blank">in</a> 
            <a href="https://github.com" target="_blank">🔗</a>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    # --- Grafik Alanı ---
    st.markdown("<br><br><br><br>", unsafe_allow_html=True) # Yukarıdan boşluk
    
    # Matplotlib grafiği (Beyaz arka plana uyarlanmış)
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.random.rand(10) * 30
    y = np.random.rand(10) * 200
    ax.scatter(x, y, color='#4CAF50', s=100) # Yeşil noktalar gibi
    
    # Grafik arka planları beyaz
    ax.set_facecolor("#FFFFFF") 
    fig.patch.set_facecolor('#FFFFFF') 
    
    # Eksen etiketleri ve çizgileri koyu
    ax.tick_params(colors='#000000') 
    ax.spines['left'].set_color('#000000') 
    ax.spines['bottom'].set_color('#000000')
    ax.spines['right'].set_color('#FFFFFF') 
    ax.spines['top'].set_color('#FFFFFF')
    
    st.pyplot(fig)

    st.markdown(
        """
        <p style='color: #FFD700; font-size: 1.5em; text-align: center; font-weight: bold;'>
            Results<br>and why
        </p>
        """, 
        unsafe_allow_html=True
    ) # Sonuçlar ve neden metni
