"""
Page 5 — UC4 : Détection d'anomalies.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render():
    from utils.ui import page_header, section_header, plotly_dark_layout
    page_header("Détection d'Anomalies", "Isolation Forest — Détection non supervisée de transactions suspectes", "⚠️")
    if 'anomaly_labels' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger vos données et entraîner les modèles (page **Données & Entraînement**).")
        st.stop()

    labels  = st.session_state['anomaly_labels']
    scores  = st.session_state['anomaly_scores']
    X_test  = st.session_state.get('X_test_raw', pd.DataFrame())

    n_total  = len(labels)
    n_anom   = int((labels == -1).sum())
    pct_anom = n_anom / n_total * 100 if n_total > 0 else 0

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Factures analysées", f"{n_total:,}")
    k2.metric("🔴 Anomalies", f"{n_anom:,}", delta=f"{pct_anom:.1f}%", delta_color="inverse")
    k3.metric("🟢 Normales", f"{n_total - n_anom:,}")
    k4.metric("Score médian anomalie", f"{np.median(scores[labels == -1]):.4f}" if n_anom > 0 else "N/A")

    # ── Distribution des scores ───────────────────────────────────────────────
    section_header("Distribution des scores d'anomalie")
    fig_hist = px.histogram(
        x=scores, color=['Anomalie' if l == -1 else 'Normal' for l in labels],
        nbins=50, barmode='overlay',
        labels={'x': "Score d'anomalie", 'color': 'Statut'},
        color_discrete_map={'Anomalie': '#ff2d78', 'Normal': '#00ff9d'},
    )
    plotly_dark_layout(fig_hist, "Distribution scores Isolation Forest")
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Scatter index / score ─────────────────────────────────────────────────
    section_header("🔍 Scores par observation")
    df_scatter = pd.DataFrame({
        'Index': range(n_total),
        'Score': scores,
        'Statut': ['🔴 Anomalie' if l == -1 else '🟢 Normal' for l in labels]
    })
    fig_scatter = px.scatter(
        df_scatter, x='Index', y='Score', color='Statut',
        color_discrete_map={'🔴 Anomalie': '#ff2d78', '🟢 Normal': '#00ff9d'},
        opacity=0.7
    )
    plotly_dark_layout(fig_scatter, "Score par observation (holdout)")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Pie chart ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(
            values=[n_anom, n_total - n_anom],
            names=['Anomalies', 'Normales'],
            color_discrete_sequence=['#e74c3c', '#2ecc71'],
            title='Répartition Anomalies / Normales'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        if not X_test.empty and 'montant_ttc' in X_test.columns:
            df_box = X_test.copy().reset_index(drop=True)
            df_box['Statut'] = ['🔴 Anomalie' if l == -1 else '🟢 Normal' for l in labels]
            fig_box = px.box(
                df_box, x='Statut', y='montant_ttc', color='Statut',
                color_discrete_map={'🔴 Anomalie': '#e74c3c', '🟢 Normal': '#2ecc71'},
                title='Montant TTC : Anomalies vs Normales'
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Boxplot non disponible (colonne `montant_ttc` absente).")

    # ── Tableau des anomalies ─────────────────────────────────────────────────
    section_header("🗂️ Détail des factures anomaliques")
    if not X_test.empty:
        anom_df = X_test.copy().reset_index(drop=True)
        anom_df['Score']  = scores.round(5)
        anom_df['Statut'] = ['🔴 Anomalie' if l == -1 else '🟢 Normal' for l in labels]
        st.dataframe(
            anom_df[anom_df['Statut'] == '🔴 Anomalie'].sort_values('Score'),
            use_container_width=True
        )
    else:
        st.dataframe(
            pd.DataFrame({'Index': np.where(labels == -1)[0], 'Score': scores[labels == -1]}),
            use_container_width=True
        )
