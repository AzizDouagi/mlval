"""
Page 4 — UC3 : Prévision de trésorerie (Prophet / trend linéaire).
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.models import forecast_cashflow, PROPHET_AVAILABLE


def render():
    from utils.ui import page_header, section_header, plotly_dark_layout
    page_header("Prévision de Trésorerie", "Modèle Prophet / régression linéaire par tenant", "📈")

    if 'df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger vos données et entraîner les modèles (page **Données & Entraînement**).")
        st.stop()

    df         = st.session_state['df']
    date_col   = st.session_state.get('date_col')
    amount_col = st.session_state.get('amount_col')

    if not date_col or not amount_col:
        st.error("❌ Colonnes date ou montant introuvables dans les données.")
        st.stop()

    method_label = "Prophet (machine learning)" if PROPHET_AVAILABLE else "Régression linéaire (Prophet non disponible)"
    st.info(f"📡 Méthode de prévision : **{method_label}**")

    tenants = sorted(df['tenant_id'].astype(str).unique().tolist())

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_tenant = st.selectbox("🏢 Sélectionner un tenant", tenants)
    with col2:
        periods = st.slider("📅 Horizon de prévision (jours)", 30, 365, 180)

    if st.button("🔮 Générer la prévision", type="primary"):
        with st.spinner("Calcul de la prévision…"):
            try:
                history, forecast = forecast_cashflow(df, selected_tenant, amount_col, date_col, periods)
                if history is None or forecast is None:
                    st.error("❌ Aucune donnée disponible pour ce tenant.")
                    st.stop()
                st.session_state['cashflow_history'] = history
                st.session_state['cashflow_forecast'] = forecast
                st.session_state['cashflow_tenant']   = selected_tenant
            except Exception as e:
                st.error(f"❌ Erreur : {e}")
                st.stop()

    if 'cashflow_history' in st.session_state:
        history  = st.session_state['cashflow_history']
        forecast = st.session_state['cashflow_forecast']
        tenant   = st.session_state['cashflow_tenant']

        forecast_future = forecast[forecast['ds'] > history['ds'].max()]

        # ── KPIs ─────────────────────────────────────────────────────────────
        k1, k2, k3 = st.columns(3)
        k1.metric("Points historiques", f"{len(history):,}")
        k2.metric("Jours prévisionnels", f"{len(forecast_future):,}")
        k3.metric("Revenu prévu total", f"{forecast_future['yhat'].sum():,.0f} TND")

        # ── Graphique principal ───────────────────────────────────────────────
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history['ds'], y=history['y'],
            name='Historique', line=dict(color='#00d4ff', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_future['ds'], y=forecast_future['yhat'],
            name='Prévision', line=dict(color='#7c3aed', width=2, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_future['ds'], forecast_future['ds'][::-1]]),
            y=pd.concat([forecast_future['yhat_upper'], forecast_future['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(124,58,237,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervalle de confiance 90%'
        ))
        plotly_dark_layout(fig, f'Prévision de trésorerie — Tenant : {tenant}', height=500)
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # ── Tableau prévisions futures ────────────────────────────────────────
        with st.expander("📋 Données prévisionnelles"):
            st.dataframe(
                forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round(2),
                use_container_width=True
            )

        # ── Agrégation mensuelle ──────────────────────────────────────────────
        section_header("📅 Prévision mensuelle agrégée")
        monthly = forecast_future.copy()
        monthly['mois'] = monthly['ds'].dt.to_period('M').astype(str)
        monthly_agg = monthly.groupby('mois')[['yhat', 'yhat_lower', 'yhat_upper']].sum().round(2).reset_index()
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(x=monthly_agg['mois'], y=monthly_agg['yhat'],
                               name='Prévision', marker_color='#7c3aed'))
        plotly_dark_layout(fig_m, 'Revenu mensuel prévu (TND)', height=400)
        st.plotly_chart(fig_m, use_container_width=True)
