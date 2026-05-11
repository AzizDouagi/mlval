"""
Page 6 — UC5 : Régression du délai de paiement.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render():
    from utils.ui import page_header, section_header, plotly_dark_layout
    page_header("Délai de Paiement", "Régression linéaire — Prédiction du nombre de jours avant paiement", "📉")

    if 'metrics_reg' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger vos données et entraîner les modèles (page **Données & Entraînement**). "
                   "La colonne `delai_paiement_j` doit être présente.")
        st.stop()

    metrics   = st.session_state['metrics_reg']
    y_test, y_pred = st.session_state['reg_data']

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("R²", f"{metrics['R2']:.4f}")
    k2.metric("RMSE (jours)", f"{metrics['RMSE']:.2f}")
    k3.metric("MAE (jours)", f"{metrics['MAE']:.2f}")

    # ── Prédictions vs Réel ───────────────────────────────────────────────────
    section_header("🎯 Prédictions vs Valeurs réelles")
    df_pred = pd.DataFrame({'Réel': y_test.values, 'Prédit': y_pred})
    fig_scatter = px.scatter(
        df_pred, x='Réel', y='Prédit',
        opacity=0.6,
        labels={'Réel': 'Délai réel (jours)', 'Prédit': 'Délai prédit (jours)'},
        color_discrete_sequence=['#00d4ff']
    )
    min_val = min(df_pred['Réel'].min(), df_pred['Prédit'].min())
    max_val = max(df_pred['Réel'].max(), df_pred['Prédit'].max())
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', name='Prédiction parfaite',
        line=dict(color='#ff2d78', dash='dash')
    ))
    plotly_dark_layout(fig_scatter, "Délai de paiement : Réel vs Prédit")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Distribution des résidus ───────────────────────────────────────────────
    section_header("📊 Distribution des résidus")
    residuals = y_test.values - y_pred
    col1, col2 = st.columns(2)
    with col1:
        fig_res = px.histogram(x=residuals, nbins=50, title='Distribution des résidus',
                               labels={'x': 'Résidu (jours)'},
                               color_discrete_sequence=['#3498db'])
        fig_res.add_vline(x=0, line_dash='dash', line_color='red')
        st.plotly_chart(fig_res, use_container_width=True)
    with col2:
        fig_qq = px.scatter(x=sorted(residuals), y=np.linspace(residuals.min(), residuals.max(), len(residuals)),
                            title='Résidus vs. Ligne de référence',
                            labels={'x': 'Résidus observés', 'y': 'Valeur de référence'})
        fig_qq.add_trace(go.Scatter(
            x=[residuals.min(), residuals.max()],
            y=[residuals.min(), residuals.max()],
            mode='lines', name='Référence', line=dict(color='red', dash='dash')
        ))
        st.plotly_chart(fig_qq, use_container_width=True)

    # ── Comparaison tabulaire ─────────────────────────────────────────────────
    section_header("📋 Échantillon prédictions")
    st.dataframe(df_pred.head(50).round(2), use_container_width=True)
