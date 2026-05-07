"""
Page 3 — UC2 : Segmentation client RFM.
"""
import streamlit as st
import pandas as pd
import plotly.express as px


def render():
    from utils.ui import page_header, section_header, plotly_dark_layout
    NEON = ['#00d4ff','#7c3aed','#00ff9d','#ff2d78','#ff6b2b','#f0db4f']
    page_header("Segmentation Client (RFM)", "Clustering K-Means par tenant — Recency / Frequency / Monetary", "👥")

    if 'rfm' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger vos données et entraîner les modèles (page **Données & Entraînement**).")
        st.stop()

    rfm: pd.DataFrame = st.session_state['rfm']

    # ── Filtre tenant ─────────────────────────────────────────────────────────
    tenants = ['Tous'] + sorted(rfm['tenant_id'].unique().tolist())
    selected = st.selectbox("🏢 Filtrer par tenant", tenants)
    rfm_view = rfm if selected == 'Tous' else rfm[rfm['tenant_id'] == selected]

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("Clients analysés", f"{len(rfm_view):,}")
    k2.metric("Tenants", f"{rfm_view['tenant_id'].nunique():,}")
    k3.metric("Segments", f"{rfm_view['Segment_Label'].nunique()}")

    # ── Distribution des segments ─────────────────────────────────────────────
    section_header("Distribution des segments")
    seg_counts = rfm_view['Segment_Label'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Nombre']
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(seg_counts, values='Nombre', names='Segment',
                         color_discrete_sequence=NEON)
        plotly_dark_layout(fig_pie, "Répartition des segments", height=360)
        fig_pie.update_traces(textfont_color='#e8f4ff')
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        fig_bar = px.bar(seg_counts, x='Segment', y='Nombre', color='Segment',
                         color_discrete_sequence=NEON)
        plotly_dark_layout(fig_bar, "Clients par segment", height=360)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── RFM par segment ───────────────────────────────────────────────────────
    section_header("Profil RFM moyen par segment")
    rfm_agg = rfm_view.groupby('Segment_Label')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
    st.dataframe(rfm_agg, use_container_width=True)

    # ── Scatter 3D RFM ────────────────────────────────────────────────────────
    section_header("🌐 Carte 3D — Recency / Frequency / Monetary")
    fig3d = px.scatter_3d(
        rfm_view, x='Recency', y='Frequency', z='Monetary',
        color='Segment_Label', opacity=0.8,
        color_discrete_sequence=NEON
    )
    fig3d.update_layout(
        scene=dict(
            bgcolor='rgba(6,11,20,0)',
            xaxis=dict(backgroundcolor='rgba(13,25,48,0.5)',gridcolor='rgba(0,212,255,0.1)',tickfont=dict(color='#7a9cc4')),
            yaxis=dict(backgroundcolor='rgba(13,25,48,0.5)',gridcolor='rgba(0,212,255,0.1)',tickfont=dict(color='#7a9cc4')),
            zaxis=dict(backgroundcolor='rgba(13,25,48,0.5)',gridcolor='rgba(0,212,255,0.1)',tickfont=dict(color='#7a9cc4')),
        ),
        paper_bgcolor='rgba(0,0,0,0)', height=580,
        legend=dict(font=dict(color='#c8dff5'), bgcolor='rgba(6,11,20,0.7)',
                    bordercolor='rgba(0,212,255,0.15)', borderwidth=1)
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # ── Scatter 2D ────────────────────────────────────────────────────────────
    section_header("Monetary vs Frequency")
    fig2d = px.scatter(rfm_view, x='Frequency', y='Monetary', color='Segment_Label',
                       size='Monetary', hover_data=['tenant_id'],
                       color_discrete_sequence=NEON)
    plotly_dark_layout(fig2d, height=420)
    st.plotly_chart(fig2d, use_container_width=True)

    # ── Tableau détaillé ──────────────────────────────────────────────────────
    section_header("🗂️ Données RFM complètes")
    st.dataframe(rfm_view.sort_values('Monetary', ascending=False).reset_index(drop=True),
                 use_container_width=True)
