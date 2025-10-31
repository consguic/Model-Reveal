# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

# Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# Optional: XGBoost if installed
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Optional: LIME
try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
except Exception:
    HAS_LIME = False

# SHAP
import shap

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Model Reveal Pro", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        pass

def load_demo_dataset(kind):
    if kind == "Classification (Iris)":
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d.frame
        return df
    if kind == "Regression (Diabetes)":
        from sklearn.datasets import load_diabetes
        d = load_diabetes(as_frame=True)
        df = d.frame
        return df
    if kind == "Anomaly (blobs+outliers)":
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=300, centers=1, n_features=2, random_state=42)
        rng = np.random.RandomState(42)
        anomalies = rng.uniform(low=-10, high=10, size=(20, 2))
        X = np.vstack([X, anomalies])
        df = pd.DataFrame(X, columns=["feat1", "feat2"])
        return df
    return None

def try_shap_explain(model, background, X_to_explain):
    """
    Try to produce a shap.Explanation object using shap.Explainer.
    Returns (explainer, shap_values) or (None, None) on failure.
    """
    try:
        # Use a small background sample to speed up kernel-like explainers
        background_sample = background.sample(min(len(background), 100))
        explainer = shap.Explainer(model, background_sample, silent=True)
        shap_values = explainer(X_to_explain)
        return explainer, shap_values
    except Exception as e:
        return None, None

def plot_shap_summary(shap_values, background_df):
    plt.figure(figsize=(6,4))
    try:
        shap.plots.beeswarm(shap_values, show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        # fallback to bar of mean abs
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1]
        feat = background_df.columns[idx]
        vals = mean_abs[idx]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(feat[:15][::-1], vals[:15][::-1])
        ax.set_title("Mean |SHAP value| (top features)")
        st.pyplot(fig)
        plt.close(fig)

def bar_feature_contributions(contrib_series, title="Feature contributions"):
    fig, ax = plt.subplots(figsize=(6,4))
    contrib_series = contrib_series.sort_values()
    ax.barh(contrib_series.index, contrib_series.values)
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

def approx_feature_deviation(test_row, background_df):
    """Approximate contribution by difference from background mean (simple heuristic)."""
    means = background_df.mean()
    diffs = test_row.squeeze() - means
    # scale by std to normalize
    stds = background_df.std().replace(0, 1)
    contrib = (diffs / stds)
    return contrib.sort_values(ascending=False)

# ---------------------------
# UI
# ---------------------------
st.title("🔍 Model Reveal Pro — Multi-Model Explainability Lab")
st.write("Sınıflandırma, regresyon ve anomali modellerini eğit, tahmin et ve kararları açıklamaya çalış.")

# Sidebar: data + task selection
with st.sidebar:
    st.header("1) Veri & Görev Seçimi")
    data_mode = st.radio("Veri kaynağı:", ("Demo veri", "Kendi CSV'im"))
    demo_choice = None
    if data_mode == "Demo veri":
        demo_choice = st.selectbox("Demo set seç:", ("Classification (Iris)", "Regression (Diabetes)", "Anomaly (blobs+outliers)"))
        df = load_demo_dataset(demo_choice)
    else:
        uploaded = st.file_uploader("CSV yükle:", type=["csv"])
        df = None
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"CSV okunamadı: {e}")

    if df is not None:
        st.success("Veri yüklendi (preview aşağıda).")
        st.dataframe(df.head(), use_container_width=True)
        # Task selection (infer if demo)
        task = st.selectbox("Problem tipi:", ("Classification", "Regression", "Anomaly")) if data_mode != "Demo veri" else (
            "Classification" if "Iris" in demo_choice else ("Regression" if "Diabetes" in demo_choice else "Anomaly")
        )
        st.markdown("---")
        st.header("2) Model Seçimi")
        if task == "Classification":
            model_choice = st.selectbox("Model:", ("RandomForestClassifier", "LogisticRegression") + (("XGBoostClassifier",) if HAS_XGB else ()))
        elif task == "Regression":
            model_choice = st.selectbox("Model:", ("RandomForestRegressor", "LinearRegression") + (("XGBRegressor",) if HAS_XGB else ()))
        else:
            model_choice = st.selectbox("Anomaly Model:", ("IsolationForest", "LocalOutlierFactor", "OneClassSVM"))

        st.markdown("---")
        st.header("3) Explainability Seçimi")
        expl_choice = st.selectbox("Tercih edilen açıklama yöntemi (öncelik):", ("SHAP (önerilen)", "LIME (varsa)", "Permutation Importance", "Feature-deviation (anomaly fallback)"))

        st.markdown("---")
        st.header("4) Eğitim / Hazırla")
        test_size = st.slider("Test boyutu (supervised için)", 0.05, 0.5, 0.2)
        random_state = st.number_input("Random seed", value=42, step=1)
        if st.button("Eğit / Hazırla"):
            st.session_state['task'] = task
            st.session_state['model_choice'] = model_choice
            st.session_state['expl_choice'] = expl_choice
            st.session_state['test_size'] = test_size
            st.session_state['random_state'] = int(random_state)
            st.session_state['raw_df'] = df.copy()
            # move along
            safe_rerun()
    else:
        st.info("Önce demo seçin veya CSV yükleyin.")

# Main area: if training requested
if 'raw_df' in st.session_state:
    raw_df = st.session_state['raw_df']
    task = st.session_state['task']
    model_choice = st.session_state['model_choice']
    expl_choice = st.session_state['expl_choice']
    test_size = st.session_state['test_size']
    random_state = st.session_state['random_state']

    st.header("📋 Veri Önizleme ve Ayarlar")
    st.write(f"Problem tipi: **{task}** — Model: **{model_choice}** — Explain: **{expl_choice}**")
    st.dataframe(raw_df.head(), use_container_width=True)

    # If supervised, ask for target column
    if task in ("Classification", "Regression"):
        possible_targets = list(raw_df.columns)
        target_col = st.selectbox("Hedef (target) sütununu seçin:", possible_targets)
        feature_df = raw_df.drop(columns=[target_col])
        y = raw_df[target_col]
        # If classification and target non-numeric, encode
        label_encoder = None
        if task == "Classification" and y.dtype == 'object':
            label_encoder = LabelEncoder()
            y_enc = label_encoder.fit_transform(y)
            st.session_state['label_encoder'] = label_encoder
        else:
            y_enc = y.values
    else:
        feature_df = raw_df.copy()
        y = None
        y_enc = None

    # Basic preprocessing: drop non-numeric columns by one-hot encoding
    X = pd.get_dummies(feature_df, drop_first=True)
    st.write(f"Özellik sayısı (after OHE): {X.shape[1]}")

    # Train/test split
    if task in ("Classification", "Regression"):
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=random_state)
    else:
        # For anomaly, use all data to fit unsupervised models; but keep background for explanations
        X_train = X.copy()
        X_test = None
        y_train = None
        y_test = None

    # Build model
    st.header("⚙️ Model Eğitimi / Fit")
    model = None
    t0 = time.time()
    try:
        if task == "Classification":
            if model_choice == "RandomForestClassifier":
                model = RandomForestClassifier(random_state=random_state, n_estimators=100)
            elif model_choice == "LogisticRegression":
                model = LogisticRegression(max_iter=1000, random_state=random_state)
            elif model_choice == "XGBoostClassifier" and HAS_XGB:
                model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)
        elif task == "Regression":
            if model_choice == "RandomForestRegressor":
                model = RandomForestRegressor(random_state=random_state, n_estimators=100)
            elif model_choice == "LinearRegression":
                model = LinearRegression()
            elif model_choice == "XGBRegressor" and HAS_XGB:
                model = XGBRegressor(random_state=random_state)
            model.fit(X_train, y_train)
        else:  # Anomaly
            if model_choice == "IsolationForest":
                model = IsolationForest(random_state=random_state, contamination='auto')
                model.fit(X_train)
            elif model_choice == "LocalOutlierFactor":
                # LOF: fit_predict returns labels; to keep API similar, fit on X and keep for predict via -1/1
                lof = LocalOutlierFactor(novelty=True)
                lof.fit(X_train)
                model = lof
            elif model_choice == "OneClassSVM":
                model = OneClassSVM(gamma='auto')
                model.fit(X_train)
    except Exception as e:
        st.error(f"Model eğitimi sırasında hata: {e}")
        st.stop()

    t1 = time.time()
    st.success(f"Model eğitildi ({t1-t0:.2f}s)")
    # Save to session for later explain/use
    st.session_state['model'] = model
    st.session_state['X_train'] = X_train
    st.session_state['X'] = X  # all processed features

    # Show basic metrics for supervised
    if task == "Classification":
        preds = model.predict(X_test)
        st.subheader("Model Performansı (Test set)")
        st.text(classification_report(y_test, preds))
        st.write("Confusion matrix:")
        st.write(confusion_matrix(y_test, preds))
    elif task == "Regression":
        preds = model.predict(X_test)
        st.subheader("Regresyon Performansı (Test set)")
        st.write(f"R2: {r2_score(y_test, preds):.3f}, RMSE: {mean_squared_error(y_test, preds, squared=False):.3f}")

    # ----------------------------
    # Explainability / Single instance analysis
    # ----------------------------
    st.header("🔬 Tek Örnek Analizi (Explain)")

    st.write("Sol taraftan tek bir örnek girebilir veya CSV ile tek satır test yükleyebilirsin.")
    col_left, col_right = st.columns([1,2])

    with col_left:
        upload_single = st.file_uploader("Tek örnek .csv (opsiyonel, tek satır):", type=["csv"], key="single_csv")
        if upload_single is not None:
            test_df = pd.read_csv(upload_single)
            test_X = pd.get_dummies(test_df, drop_first=True).reindex(columns=X_train.columns, fill_value=0)
            st.session_state['test_X'] = test_X
        elif st.button("Rastgele Bir Test Örneği Seç"):
            if task in ("Classification", "Regression"):
                # choose random from X_test
                idx = np.random.choice(X_test.shape[0], 1)
                test_X = X_test.iloc[idx]
            else:
                idx = np.random.choice(X_train.shape[0], 1)
                test_X = X_train.iloc[idx]
            st.session_state['test_X'] = test_X
            st.write("Rastgele örnek seçildi.")
        else:
            st.info("Tek satır test yükle veya rastgele örnek seç.")

        st.markdown("---")
        st.write("Explain yöntemi tercihi:", expl_choice)
        st.write("SHAP çalışmazsa otomatik fallback uygulanacaktır.")

    with col_right:
        if 'test_X' in st.session_state:
            test_X = st.session_state['test_X']
            st.subheader("Test örneği (işlenmiş)")
            st.dataframe(test_X.T)
            # Prediction
            model = st.session_state['model']
            try:
                if task in ("Classification", "Regression"):
                    pred = model.predict(test_X)[0]
                    proba = model.predict_proba(test_X)[0] if hasattr(model, "predict_proba") else None
                    st.metric("Tahmin", str(pred))
                    if proba is not None:
                        st.metric("Confidence", f"{max(proba)*100:.2f}%")
                else:
                    pred = model.predict(test_X)[0]
                    label = "Anomali" if pred == -1 else "Normal"
                    score = model.decision_function(test_X)[0] if hasattr(model, "decision_function") else None
                    st.metric("Tespit", label)
                    if score is not None:
                        st.metric("Anomali skoru", f"{score:.4f}")
            except Exception as e:
                st.error(f"Tahmin yapılamadı: {e}")

            st.subheader("Açıklama (Explainability)")

            explained = False
            # Try SHAP if selected
            if expl_choice.startswith("SHAP"):
                explainer, shap_values = try_shap_explain(model, st.session_state['X_train'], test_X)
                if explainer is not None and shap_values is not None:
                    st.success("SHAP ile açıklama üretildi.")
                    # show waterfall for single instance if available
                    try:
                        # shap_values may be an Explanation object
                        if hasattr(shap_values, "values"):
                            # For classification, shap_values.shape may be (n_instances, n_features) or list
                            plot_shap_summary(shap_values, st.session_state['X_train'])
                        else:
                            plot_shap_summary(shap_values, st.session_state['X_train'])
                        explained = True
                    except Exception as e:
                        st.warning(f"SHAP görselleştirme hatası: {e}")
                else:
                    st.warning("SHAP açıklaması oluşturulamadı veya model desteklemiyor. Fallback uygulanacak.")

            # If LIME selected and available
            if not explained and expl_choice.startswith("LIME") and HAS_LIME:
                try:
                    explainer_lime = LimeTabularExplainer(st.session_state['X_train'].values,
                                                          feature_names=st.session_state['X_train'].columns.tolist(),
                                                          class_names=None if task!="Classification" else ["class"],
                                                          discretize_continuous=True)
                    exp = explainer_lime.explain_instance(test_X.values[0], model.predict_proba if task=="Classification" else model.predict, num_features=10)
                    st.subheader("LIME explanation (top features)")
                    st.write(dict(exp.as_list()))
                    explained = True
                except Exception as e:
                    st.warning(f"LIME hata: {e}")

            # Permutation importance fallback
            if not explained and expl_choice.startswith("Permutation"):
                try:
                    if task in ("Classification", "Regression"):
                        res = permutation_importance(model, X_test if task!="Anomaly" else st.session_state['X_train'], y_test if task!="Anomaly" else None, n_repeats=10, random_state=0, n_jobs=1)
                        imp = pd.Series(res.importances_mean, index=st.session_state['X_train'].columns).sort_values(ascending=False)
                        st.subheader("Permutation Importance (ortalama)")
                        bar_feature_contributions(imp, title="Permutation Importance")
                        explained = True
                    else:
                        st.warning("Permutation importance anomali modelleri için uygun değildir. Feature-deviation fallback uygulanacak.")
                except Exception as e:
                    st.warning(f"Permutation hesaplanırken hata: {e}")

            # Anomaly / feature-deviation fallback
            if not explained:
                st.info("Feature-deviation yöntemi ile kaba katkı tahmini gösteriliyor.")
                try:
                    contrib = approx_feature_deviation(test_X, st.session_state['X_train'])
                    st.subheader("Yaklaşık Özellik Katkıları (z-score farkı)")
                    bar_feature_contributions(contrib, title="Approx Feature deviation (z-score)")
                    explained = True
                except Exception as e:
                    st.error(f"Fallback açıklama hesaplanamadı: {e}")

            st.markdown("---")
            if st.button("Yeni Analiz (Reset)"):
                for k in ['raw_df','model','X_train','X','test_X','label_encoder']:
                    if k in st.session_state:
                        del st.session_state[k]
                safe_rerun()

        else:
            st.info("Önce tek örnek yükleyin ya da rastgele örnek seçin.")
