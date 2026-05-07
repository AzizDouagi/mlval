"""
Page 7 — UC6 : Catégorisation NLP des dépenses.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.models import predict_nlp_category


def render():
    from utils.ui import page_header, section_header, plotly_dark_layout
    page_header("Catégorisation NLP des Dépenses", "TF-IDF / Sentence Transformers + Logistic Regression", "🧾")

    # ── Résultats d'entraînement ───────────────────────────────────────────────
    if 'acc_nlp' in st.session_state:
        mode   = st.session_state.get('mode_nlp', 'N/A')
        acc    = st.session_state['acc_nlp']
        report = st.session_state.get('report_nlp', {})

        k1, k2 = st.columns(2)
        k1.metric("Accuracy (holdout tenant-aware)", f"{acc:.4f}")
        k2.metric("Mode d'embedding", mode)

        if report:
            section_header("Rapport de classification")
            rows = []
            for label, vals in report.items():
                if label in ('accuracy', 'macro avg', 'weighted avg'):
                    continue
                if isinstance(vals, dict):
                    rows.append({
                        'Catégorie': label,
                        'Precision': round(vals.get('precision', 0), 4),
                        'Recall': round(vals.get('recall', 0), 4),
                        'F1-score': round(vals.get('f1-score', 0), 4),
                        'Support': int(vals.get('support', 0))
                    })
            if rows:
                df_report = pd.DataFrame(rows)
                st.dataframe(df_report, use_container_width=True, hide_index=True)

                fig = px.bar(df_report, x='Catégorie', y='F1-score', color='Catégorie',
                             color_discrete_sequence=['#00d4ff','#7c3aed','#00ff9d','#ff2d78','#ff6b2b','#f0db4f'])
                plotly_dark_layout(fig, "F1-score par catégorie")
                st.plotly_chart(fig, use_container_width=True)

            # Macro / Weighted avg
            if 'macro avg' in report:
                st.markdown("**Macro average :**")
                st.json({k: round(v, 4) for k, v in report['macro avg'].items() if k != 'support'})
    else:
        st.info("ℹ️ Les métriques d'entraînement NLP ne sont pas disponibles. Entraînez d'abord les modèles.")

    # ── Prédiction en temps réel ──────────────────────────────────────────────
    st.markdown("")
    section_header("🔮 Prédire la catégorie d'une description")

    with st.form("nlp_form"):
        description = st.text_area(
            "📝 Description de la dépense",
            placeholder="Ex: achat de fournitures de bureau, abonnement logiciel, maintenance équipement...",
            height=100
        )
        extras = st.columns(3)
        mode_pai  = extras[0].selectbox("Mode de paiement", ['', 'virement', 'chèque', 'espèces', 'carte'])
        ville_val = extras[1].text_input("Ville", placeholder="Ex: Tunis")
        regime    = extras[2].selectbox("Régime TVA", ['', 'normal', 'forfait', 'exonéré'])
        submitted = st.form_submit_button("🧠 Prédire la catégorie", type="primary")

    if submitted and description.strip():
        text_parts = [description]
        if mode_pai:   text_parts.append(f"mode_paiement:{mode_pai}")
        if ville_val:  text_parts.append(f"ville:{ville_val}")
        if regime:     text_parts.append(f"regime_tva:{regime}")
        full_text = ' '.join(text_parts)

        result = predict_nlp_category(full_text)
        st.success(f"🏷️ Catégorie prédite : **{result}**")

    elif submitted:
        st.warning("⚠️ Veuillez saisir une description.")

    # ── Prédiction batch ─────────────────────────────────────────────────────
    st.markdown("")
    section_header("📦 Prédiction en lot (batch)")
    if 'df' in st.session_state:
        df = st.session_state['df']
        if 'description' in df.columns:
            n_sample = st.slider("Nombre de factures à analyser", 10, min(500, len(df)), 50)
            if st.button("Lancer la prédiction batch"):
                sample = df.head(n_sample).copy()
                meta_cols = [c for c in ['mode_paiement', 'ville', 'regime_tva', 'montant_bucket', 'facture_trimestre']
                             if c in sample.columns]
                def build_text(row):
                    parts = [str(row.get('description', '') or '')]
                    for m in meta_cols:
                        v = row.get(m)
                        if pd.notna(v) and str(v).strip():
                            parts.append(f'{m}:{v}')
                    return ' '.join(parts)
                sample['nlp_input'] = sample.apply(build_text, axis=1)
                with st.spinner("Prédictions en cours…"):
                    sample['Catégorie prédite'] = sample['nlp_input'].apply(predict_nlp_category)
                cols_show = ['description'] + meta_cols + ['Catégorie prédite']
                if 'categorie_produit' in sample.columns:
                    cols_show.insert(-1, 'categorie_produit')
                cols_show = [c for c in cols_show if c in sample.columns]
                st.dataframe(sample[cols_show], use_container_width=True)

                # Distribution des catégories prédites
                cat_dist = sample['Catégorie prédite'].value_counts().reset_index()
                cat_dist.columns = ['Catégorie', 'Nombre']
                fig_cat = px.bar(cat_dist, x='Catégorie', y='Nombre', color='Catégorie',
                                 color_discrete_sequence=['#00d4ff','#7c3aed','#00ff9d','#ff2d78','#ff6b2b','#f0db4f'])
                plotly_dark_layout(fig_cat, "Distribution des catégories prédites")
                st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("ℹ️ Colonne `description` absente du dataset — prédiction batch indisponible.")
    else:
        st.info("ℹ️ Chargez d'abord un dataset pour la prédiction batch.")
