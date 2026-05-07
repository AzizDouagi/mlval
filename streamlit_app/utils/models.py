"""
Model training utilities — one function per use case.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib, pickle, os

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, r2_score,
    mean_absolute_error, mean_squared_error,
    classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


def _ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# USE CASE 1 — Payment Risk Classification
# ─────────────────────────────────────────────
def train_payment_risk(X_train, X_test, y_train, y_test, groups_train):
    from sklearn.model_selection import GroupKFold, cross_val_score

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    }
    metrics = {}
    preds = {}
    probas = {}
    n_splits = min(5, groups_train.nunique())
    cv = GroupKFold(n_splits=n_splits)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        preds[name] = y_pred
        probas[name] = y_proba
        cv_score = cross_val_score(model, X_train, y_train, cv=cv, groups=groups_train, scoring='f1').mean()
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan
        metrics[name] = {
            'CV F1': cv_score,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': auc
        }

    best_name = max(metrics, key=lambda k: metrics[k]['ROC-AUC'] if not np.isnan(metrics[k]['ROC-AUC']) else -1)
    best_model = models[best_name]

    _ensure_models_dir()
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'payment_risk.pkl'))

    return models, metrics, preds, probas, best_name


def predict_risk(X_processed):
    path = os.path.join(MODELS_DIR, 'payment_risk.pkl')
    if not os.path.exists(path):
        return None
    model = joblib.load(path)
    return model.predict_proba(X_processed)[:, 1]


# ─────────────────────────────────────────────
# USE CASE 2 — RFM Segmentation
# ─────────────────────────────────────────────
def compute_rfm_segments(df, customer_col, amount_col, date_col):
    df = df.copy()
    df['_invoice_date'] = pd.to_datetime(df[date_col], errors='coerce')
    latest_date = df['_invoice_date'].max()

    rfm = (
        df.groupby(['tenant_id', customer_col])
        .agg(
            Recency=('_invoice_date', lambda x: int((latest_date - x.max()).days)),
            Frequency=(customer_col, 'count'),
            Monetary=(amount_col, 'sum')
        )
        .reset_index()
    )
    rfm['Monetary'] = rfm['Monetary'].astype(float)

    segments = []
    last_scaler = None
    last_kmeans = None
    for tenant, tg in rfm.groupby('tenant_id'):
        if len(tg) < 2:
            tg = tg.copy()
            tg['Segment'] = -1
            tg['Segment_Label'] = 'Insufficient Data'
            segments.append(tg)
            continue
        scaler = StandardScaler()
        values = scaler.fit_transform(tg[['Recency', 'Frequency', 'Monetary']])
        k = min(4, len(tg))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        tg = tg.copy()
        tg['Segment'] = km.fit_predict(values)
        summary = (
            tg.groupby('Segment')[['Monetary', 'Frequency', 'Recency']]
            .mean()
            .sort_values(['Monetary', 'Frequency', 'Recency'], ascending=[False, False, True])
        )
        labels = ['Champions', 'Loyal', 'At Risk', 'Lost'][:len(summary)]
        mapping = {seg: labels[i] for i, seg in enumerate(summary.index)}
        tg['Segment_Label'] = tg['Segment'].map(mapping)
        last_scaler = scaler
        last_kmeans = km
        segments.append(tg)

    rfm_result = pd.concat(segments, ignore_index=True)
    _ensure_models_dir()
    if last_kmeans is not None:
        joblib.dump(last_kmeans, os.path.join(MODELS_DIR, 'kmeans_rfm.pkl'))
    if last_scaler is not None:
        joblib.dump(last_scaler, os.path.join(MODELS_DIR, 'rfm_scaler.pkl'))
    return rfm_result


# ─────────────────────────────────────────────
# USE CASE 3 — Cashflow Forecasting
# ─────────────────────────────────────────────
def forecast_cashflow(df, tenant_id, amount_col, date_col, periods=180):
    from datetime import timedelta
    df = df.copy()
    df['_date_ts'] = pd.to_datetime(df[date_col], errors='coerce')
    cashflow = (
        df[df['tenant_id'] == tenant_id]
        .groupby('_date_ts')[amount_col]
        .sum()
        .reset_index()
        .rename(columns={'_date_ts': 'ds', amount_col: 'y'})
        .sort_values('ds')
    )
    if cashflow.empty:
        return None, None

    if PROPHET_AVAILABLE:
        m = Prophet(yearly_seasonality=True, daily_seasonality=False,
                    interval_width=0.90, seasonality_mode='additive')
        m.fit(cashflow)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        _ensure_models_dir()
        with open(os.path.join(MODELS_DIR, 'prophet_model.pkl'), 'wb') as f:
            pickle.dump(m, f)
    else:
        x = np.arange(len(cashflow))
        slope, intercept = np.polyfit(x, cashflow['y'].values, 1)
        residuals = cashflow['y'].values - (intercept + slope * x)
        std_err = np.std(residuals)
        future_dates = pd.date_range(
            start=cashflow['ds'].max() + timedelta(days=1), periods=periods, freq='D'
        )
        forecast = pd.DataFrame({
            'ds': future_dates,
            'yhat': intercept + slope * np.arange(len(cashflow), len(cashflow) + periods),
            'yhat_lower': intercept + slope * np.arange(len(cashflow), len(cashflow) + periods) - 1.96 * std_err,
            'yhat_upper': intercept + slope * np.arange(len(cashflow), len(cashflow) + periods) + 1.96 * std_err
        })

    return cashflow, forecast


# ─────────────────────────────────────────────
# USE CASE 4 — Anomaly Detection
# ─────────────────────────────────────────────
def train_anomaly_detector(X_train_processed, X_test_processed, preprocessor):
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_train_processed)
    labels = iso.predict(X_test_processed)
    scores = iso.score_samples(X_test_processed)
    _ensure_models_dir()
    joblib.dump(iso, os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    return iso, labels, scores


def predict_anomalies(X_processed):
    path = os.path.join(MODELS_DIR, 'isolation_forest.pkl')
    if not os.path.exists(path):
        return None
    model = joblib.load(path)
    return model.predict(X_processed), model.score_samples(X_processed)


# ─────────────────────────────────────────────
# USE CASE 5 — Payment Delay Regression
# ─────────────────────────────────────────────
def train_delay_regression(df, train_idx, test_idx):
    if 'delai_paiement_j' not in df.columns:
        return None, None, None

    y_reg = df['delai_paiement_j'].astype(float).fillna(df['delai_paiement_j'].median())
    leakage = {
        'retard_paiement', 'delai_paiement_j', 'description', 'description_enhanced',
        'categorie_produit', 'tenant_id', 'invoice_id', 'client_id', 'client_nom',
        'date_facture', 'date_echeance', 'statut_facture', 'facture_date'
    }
    leakage |= {c for c in df.columns if 'date' in c.lower() and
                any(tok in c.lower() for tok in ['facture', 'echeance', 'paye'])}

    reg_features = [c for c in df.columns if c not in leakage]
    if not reg_features:
        return None, None, None

    X_reg = df[reg_features].copy()
    X_reg_train = X_reg.iloc[train_idx].reset_index(drop=True)
    X_reg_test = X_reg.iloc[test_idx].reset_index(drop=True)
    y_train = y_reg.iloc[train_idx].reset_index(drop=True)
    y_test = y_reg.iloc[test_idx].reset_index(drop=True)

    reg_num = X_reg.select_dtypes(include=[np.number]).columns.tolist()
    reg_cat = X_reg.select_dtypes(include=['object', 'category']).columns.tolist()

    reg_preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), reg_num),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='unknown')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))]), reg_cat)
    ])
    pipeline = Pipeline([('preprocessor', reg_preprocessor), ('regressor', LinearRegression())])
    pipeline.fit(X_reg_train, y_train)
    y_pred = pipeline.predict(X_reg_test)

    metrics = {
        'R2': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    _ensure_models_dir()
    joblib.dump(pipeline, os.path.join(MODELS_DIR, 'delay_regression.pkl'))
    return pipeline, metrics, (y_test, y_pred)


# ─────────────────────────────────────────────
# USE CASE 6 — NLP Expense Categorization
# ─────────────────────────────────────────────
def train_nlp_categorization(df, train_idx, test_idx):
    if 'categorie_produit' not in df.columns:
        return None, None, None, None

    meta_cols = [c for c in ['mode_paiement', 'ville', 'regime_tva', 'montant_bucket', 'facture_trimestre']
                 if c in df.columns]

    def build_text(row):
        parts = [str(row.get('description', '') or '')]
        for m in meta_cols:
            v = row.get(m)
            if pd.notna(v) and str(v).strip():
                parts.append(f'{m}:{v}')
        return ' '.join(parts)

    df = df.copy()
    df['nlp_input'] = df.apply(build_text, axis=1)
    y_nlp = df['categorie_produit'].astype(str).fillna('unknown')

    if EMBEDDING_AVAILABLE:
        emb_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        X_nlp = emb_model.encode(df['nlp_input'].tolist(), show_progress_bar=False)
        mode = 'sentence-transformers'
    else:
        tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')
        X_nlp = tfidf.fit_transform(df['nlp_input']).toarray()
        mode = 'tfidf'
        joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))

    X_train = X_nlp[train_idx]
    X_test = X_nlp[test_idx]
    y_train = y_nlp.iloc[train_idx].reset_index(drop=True)
    y_test = y_nlp.iloc[test_idx].reset_index(drop=True)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    acc = accuracy_score(y_test, y_pred)

    _ensure_models_dir()
    joblib.dump(clf, os.path.join(MODELS_DIR, 'nlp_categories.pkl'))
    return clf, mode, acc, report


def predict_nlp_category(text: str) -> str:
    clf_path = os.path.join(MODELS_DIR, 'nlp_categories.pkl')
    tfidf_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    if not os.path.exists(clf_path):
        return 'Modèle non entraîné'
    clf = joblib.load(clf_path)
    if os.path.exists(tfidf_path):
        tfidf = joblib.load(tfidf_path)
        X = tfidf.transform([text]).toarray()
    else:
        if EMBEDDING_AVAILABLE:
            emb = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            X = emb.encode([text])
        else:
            return 'Vectoriseur non disponible'
    return clf.predict(X)[0]
