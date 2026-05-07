"""
Shared UI helpers — futuristic components.
"""
import streamlit as st


def page_header(title: str, subtitle: str = "", icon: str = ""):
    """Render a styled futuristic page header."""
    st.markdown(f"""
    <div style="margin-bottom:1.8rem;">
        <h1 style="
            font-family:'Space Grotesk',sans-serif;
            font-weight:900;font-size:2rem;
            background:linear-gradient(135deg,#00d4ff 0%,#7c3aed 100%);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;letter-spacing:-0.02em;margin:0 0 0.3rem 0;
        ">{icon} {title}</h1>
        {"<p style='color:#7a9cc4;font-size:0.92rem;margin:0;'>" + subtitle + "</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.7rem;margin:1.5rem 0 0.8rem 0;">
        <div style="width:3px;height:1.2rem;border-radius:2px;
                    background:linear-gradient(135deg,#00d4ff,#7c3aed);"></div>
        <h3 style="margin:0;font-family:'Space Grotesk',sans-serif;
                   color:#e8f4ff;font-size:1.05rem;font-weight:700;">{title}</h3>
    </div>
    """, unsafe_allow_html=True)


def stat_card(label: str, value: str, color: str = "#00d4ff", delta: str = ""):
    delta_html = f"<div style='font-size:0.72rem;color:{color};margin-top:0.1rem;'>{delta}</div>" if delta else ""
    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,rgba(0,212,255,0.05),rgba(124,58,237,0.05));
        border:1px solid rgba(0,212,255,0.18);border-radius:14px;
        padding:1.1rem 1.3rem;transition:all 0.3s ease;
        box-shadow:0 2px 12px rgba(0,0,0,0.4);">
        <div style="font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:#7a9cc4;margin-bottom:0.4rem;">{label}</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.55rem;
                    font-weight:700;color:{color};">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def info_banner(text: str, color: str = "#00d4ff"):
    st.markdown(f"""
    <div style="
        background:rgba(0,0,0,0.2);
        border-left:3px solid {color};
        border-radius:0 10px 10px 0;
        padding:0.7rem 1rem;
        color:#c8dff5;font-size:0.88rem;
        margin:0.5rem 0 1rem 0;">
        {text}
    </div>
    """, unsafe_allow_html=True)


def plotly_dark_layout(fig, title: str = "", height: int = 450):
    """Apply consistent dark theme to a plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Grotesk", size=14, color="#e8f4ff")),
        paper_bgcolor="rgba(6,11,20,0)",
        plot_bgcolor="rgba(13,25,48,0.5)",
        font=dict(family="Inter", color="#7a9cc4", size=11),
        height=height,
        margin=dict(t=50 if title else 20, r=20, b=40, l=50),
        xaxis=dict(gridcolor="rgba(0,212,255,0.08)", linecolor="rgba(0,212,255,0.15)",
                   tickfont=dict(color="#7a9cc4")),
        yaxis=dict(gridcolor="rgba(0,212,255,0.08)", linecolor="rgba(0,212,255,0.15)",
                   tickfont=dict(color="#7a9cc4")),
        legend=dict(bgcolor="rgba(6,11,20,0.7)", bordercolor="rgba(0,212,255,0.15)",
                    borderwidth=1, font=dict(color="#c8dff5")),
        colorway=["#00d4ff", "#7c3aed", "#00ff9d", "#ff2d78", "#ff6b2b", "#f0db4f"]
    )
    return fig
