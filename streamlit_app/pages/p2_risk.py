"""
Page 2 — UC1 : Risque de paiement (classification).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render():
    from utils.ui import page_header, section_header, plotly_dark_layout
    page_header("Risque de Paiement", "Classification tenant-aware — Logistic Regression & Random Forest", "💰")

    if 'metrics_uc1' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger vos données et entraîner les modèles (page **Données & Entraînement**).")
        st.stop()

    metrics = st.session_state['metrics_uc1']
    best    = st.session_state['best_risk_name']

    # ── KPIs ──────────────────────────────────────────────────────────────────
    section_header("Performances des modèles")
    cols = st.columns(len(metrics))
    for i, (name, m) in enumerate(metrics.items()):
        with cols[i]:
            crown = "👑 " if name == best else ""
            st.metric(f"{crown}{name}", f"{m['ROC-AUC']:.4f}",
                      delta=f"F1: {m['F1']:.3f}")
            st.caption(f"Acc: {m['Accuracy']:.3f} · P: {m['Precision']:.3f} · R: {m['Recall']:.3f}")

    # ── Radar + Bar ───────────────────────────────────────────────────────────
    section_header("Comparaison des métriques")
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    col_left, col_right = st.columns(2)

    with col_left:
        fig = go.Figure()
        colors = ['#00d4ff', '#7c3aed', '#00ff9d', '#ff2d78']
        for i, (name, m) in enumerate(metrics.items()):
            values = [m[k] if not np.isnan(m[k]) else 0 for k in metric_names]
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=metric_names + [metric_names[0]],
                fill='toself', name=name,
                line=dict(color=colors[i % len(colors)], width=2),
                fillcolor=colors[i % len(colors)].replace('#', 'rgba(').replace(')', ',0.12)') if False else None
            ))
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(13,25,48,0.5)',
                radialaxis=dict(visible=True, range=[0,1],
                                gridcolor='rgba(0,212,255,0.1)',
                                tickfont=dict(color='#7a9cc4', size=9)),
                angularaxis=dict(gridcolor='rgba(0,212,255,0.1)',
                                 tickfont=dict(color='#e8f4ff', size=10))
            ),
            paper_bgcolor='rgba(0,0,0,0)', showlegend=True,
            legend=dict(font=dict(color='#c8dff5'), bgcolor='rgba(6,11,20,0.7)',
                        bordercolor='rgba(0,212,255,0.15)', borderwidth=1),
            height=380, margin=dict(t=30,r=20,b=20,l=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        rows = []
        for name, m in metrics.items():
            for metric in metric_names:
                rows.append({'Modèle': name, 'Métrique': metric,
                             'Score': m[metric] if not np.isnan(m[metric]) else 0})
        df_m = pd.DataFrame(rows)
        fig2 = px.bar(df_m, x='Métrique', y='Score', color='Modèle', barmode='group',
                      range_y=[0,1],
                      color_discrete_sequence=['#00d4ff','#7c3aed','#00ff9d','#ff2d78'])
        plotly_dark_layout(fig2, "Scores détaillés", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tableau récapitulatif ─────────────────────────────────────────────────
    section_header("Tableau récapitulatif")
    df_table = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Modèle'})
    df_table = df_table.round(4)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ── Résultats holdout ─────────────────────────────────────────────────────
    st.markdown("---")
    section_header("Résultats sur le holdout")
    if 'df' in st.session_state and 'anomaly_scores' in st.session_state:
        df = st.session_state['df']
        X_test_raw = st.session_state.get('X_test_raw', None)
        labels = st.session_state.get('anomaly_labels', None)
        scores = st.session_state.get('anomaly_scores', None)
        if X_test_raw is not None and labels is not None:
            display_df = X_test_raw.copy().reset_index(drop=True)
            display_df['Anomalie'] = ['🔴 Oui' if l == -1 else '🟢 Non' for l in labels]
            display_df['Score Anomalie'] = scores.round(4)
            st.dataframe(display_df.head(100), use_container_width=True)
