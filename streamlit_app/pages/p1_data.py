"""
Page 1 — Chargement des données & entraînement de tous les modèles.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.preprocessing import build_tenant_id, feature_engineering, build_feature_matrix, build_preprocessor
from utils.models import (
    train_payment_risk, compute_rfm_segments, train_anomaly_detector,
    train_delay_regression, train_nlp_categorization, PROPHET_AVAILABLE, EMBEDDING_AVAILABLE
)
from sklearn.model_selection import GroupShuffleSplit


def render():
    from utils.ui import page_header, section_header, info_banner
    page_header(
        "Données & Entraînement",
        "Chargez votre fichier CSV de factures pour entraîner tous les modèles en un clic.",
        "📂"
    )

    # ── Upload ────────────────────────────────────────────────────────────────
    default_path = BASE_DIR / 'data' / 'invoices_ml (1).csv'
    upload = st.file_uploader("📎 Importer un fichier CSV", type=['csv'])

    if upload:
        df_raw = pd.read_csv(upload, encoding='utf-8')
        st.success(f"✅ Fichier chargé : {df_raw.shape[0]:,} lignes × {df_raw.shape[1]} colonnes")
    elif default_path.exists():
        df_raw = pd.read_csv(default_path, encoding='utf-8')
        st.info(f"📂 Fichier local détecté : `data/invoices_ml (1).csv` — {df_raw.shape[0]:,} lignes")
    else:
        st.warning("⚠️ Aucun fichier CSV disponible. Importez votre fichier ci-dessus.")
        st.stop()

    with st.expander("🔍 Aperçu des données brutes", expanded=False):
        st.dataframe(df_raw.head(20), use_container_width=True)
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            st.markdown("**Valeurs manquantes :**")
            st.dataframe(
                pd.DataFrame({'Colonne': missing.index, 'Manquants': missing.values,
                              '%': (missing.values / len(df_raw) * 100).round(2)}),
                use_container_width=True
            )

    # ── Preprocessing ─────────────────────────────────────────────────────────
    st.markdown("")
    if st.button("⚡ Lancer le prétraitement & l'entraînement de tous les modèles", type="primary", use_container_width=True):
        with st.spinner("Prétraitement en cours…"):
            try:
                df, tenant_col = build_tenant_id(df_raw)
                df, date_col = feature_engineering(df)
                X, y = build_feature_matrix(df)
                preprocessor, num_feats, cat_feats = build_preprocessor(X)

                groups = df['tenant_id'].astype(str)
                if groups.nunique() < 2:
                    st.error("❌ Au moins 2 tenants distincts sont requis.")
                    st.stop()

                gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
                train_idx, test_idx = next(gss.split(X, y, groups=groups))

                X_train = X.iloc[train_idx].reset_index(drop=True)
                X_test  = X.iloc[test_idx].reset_index(drop=True)
                y_train = y.iloc[train_idx].reset_index(drop=True)
                y_test  = y.iloc[test_idx].reset_index(drop=True)
                groups_train = groups.iloc[train_idx].reset_index(drop=True)

                X_train_p = preprocessor.fit_transform(X_train)
                X_test_p  = preprocessor.transform(X_test)

                st.success(f"✅ Prétraitement OK — Train: {X_train.shape[0]:,} / Test: {X_test.shape[0]:,}")

            except Exception as e:
                st.error(f"❌ Erreur prétraitement : {e}")
                st.stop()

        # ── UC1 ──────────────────────────────────────────────────────────────
        with st.spinner("🏋️ Entraînement UC1 — Risque de paiement…"):
            try:
                models_uc1, metrics_uc1, preds_uc1, probas_uc1, best_name = train_payment_risk(
                    X_train_p, X_test_p, y_train, y_test, groups_train
                )
                st.session_state['metrics_uc1'] = metrics_uc1
                st.session_state['best_risk_name'] = best_name
                st.success(f"✅ UC1 — Meilleur modèle : **{best_name}** | ROC-AUC: {metrics_uc1[best_name]['ROC-AUC']:.4f}")
            except Exception as e:
                st.warning(f"⚠️ UC1 ignoré : {e}")

        # ── UC2 ──────────────────────────────────────────────────────────────
        with st.spinner("🏋️ Entraînement UC2 — Segmentation RFM…"):
            try:
                customer_col = next((c for c in df.columns if 'client' in c.lower() and 'id' in c.lower()), None)
                amount_col   = next((c for c in df.columns if any(t in c.lower() for t in ['montant', 'amount', 'total', 'prix'])), None)
                if customer_col and amount_col and date_col:
                    rfm = compute_rfm_segments(df, customer_col, amount_col, date_col)
                    st.session_state['rfm'] = rfm
                    st.success(f"✅ UC2 — Segmentation RFM sur {rfm['tenant_id'].nunique()} tenants")
                else:
                    st.warning("⚠️ UC2 ignoré : colonnes client_id / montant / date manquantes.")
            except Exception as e:
                st.warning(f"⚠️ UC2 ignoré : {e}")

        # ── UC3 (données stockées pour la page dédiée) ────────────────────────
        st.session_state['df']          = df
        st.session_state['date_col']    = date_col
        st.session_state['amount_col']  = next((c for c in df.columns if any(t in c.lower() for t in ['montant', 'amount', 'total', 'prix'])), None)
        st.session_state['customer_col']= next((c for c in df.columns if 'client' in c.lower() and 'id' in c.lower()), None)

        # ── UC4 ──────────────────────────────────────────────────────────────
        with st.spinner("🏋️ Entraînement UC4 — Détection d'anomalies…"):
            try:
                iso, labels, scores = train_anomaly_detector(X_train_p, X_test_p, preprocessor)
                n_anom = int((labels == -1).sum())
                st.session_state['anomaly_labels'] = labels
                st.session_state['anomaly_scores'] = scores
                st.session_state['X_test_raw'] = X_test
                st.success(f"✅ UC4 — {n_anom} anomalies détectées ({n_anom/len(X_test_p)*100:.1f}%)")
            except Exception as e:
                st.warning(f"⚠️ UC4 ignoré : {e}")

        # ── UC5 ──────────────────────────────────────────────────────────────
        with st.spinner("🏋️ Entraînement UC5 — Régression délai de paiement…"):
            try:
                pipeline_reg, metrics_reg, reg_data = train_delay_regression(df, train_idx, test_idx)
                if pipeline_reg:
                    st.session_state['metrics_reg'] = metrics_reg
                    st.session_state['reg_data'] = reg_data
                    st.success(f"✅ UC5 — R²: {metrics_reg['R2']:.4f} | MAE: {metrics_reg['MAE']:.2f} j")
                else:
                    st.warning("⚠️ UC5 ignoré : colonne `delai_paiement_j` manquante.")
            except Exception as e:
                st.warning(f"⚠️ UC5 ignoré : {e}")

        # ── UC6 ──────────────────────────────────────────────────────────────
        with st.spinner("🏋️ Entraînement UC6 — Catégorisation NLP…"):
            try:
                clf_nlp, mode_nlp, acc_nlp, report_nlp = train_nlp_categorization(df, train_idx, test_idx)
                if clf_nlp:
                    st.session_state['mode_nlp'] = mode_nlp
                    st.session_state['acc_nlp']  = acc_nlp
                    st.session_state['report_nlp'] = report_nlp
                    st.success(f"✅ UC6 — Accuracy: {acc_nlp:.4f} (mode: {mode_nlp})")
                else:
                    st.warning("⚠️ UC6 ignoré : colonne `categorie_produit` manquante.")
            except Exception as e:
                st.warning(f"⚠️ UC6 ignoré : {e}")

        st.session_state['trained'] = True
        st.balloons()
        st.success("🎉 Tous les modèles ont été entraînés et sauvegardés dans `models/`")

    # ── Statut rapide ──────────────────────────────────────────────────────────
    if st.session_state.get('trained'):
        st.markdown("---")
        section_header = __import__('utils.ui', fromlist=['section_header']).section_header
        section_header("Résumé des modèles entraînés")
        rows = []
        for uc, key, label in [
            ('UC1 – Risque paiement', 'metrics_uc1', 'ROC-AUC'),
            ('UC2 – Segmentation RFM', 'rfm', 'Segments'),
            ('UC4 – Anomalies', 'anomaly_labels', 'Labels'),
            ('UC5 – Délai paiement', 'metrics_reg', 'R2'),
            ('UC6 – NLP Catégories', 'acc_nlp', 'Accuracy'),
        ]:
            status = '✅ Prêt' if key in st.session_state else '⚠️ Ignoré'
            rows.append({'Use Case': uc, 'Statut': status})
        uc3_status = '✅ Prêt' if st.session_state.get('df') is not None else '⚠️ Ignoré'
        rows.insert(2, {'Use Case': 'UC3 – Prévision trésorerie', 'Statut': uc3_status})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
