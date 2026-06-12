"""
⚽ World Cup Intelligence Platform — v2
ML Engineer · Analyse historique FIFA 1930–2014
Navigation par boutons en haut de page + tous bugs corrigés
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report
from data_engine import (
    build_dataframes, build_team_stats, build_clusters,
    build_ml_model, predict_match, PROFIL_ORDER
)
 
# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="World Cup Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
GOLD   = "#F0C040"
BLUE   = "#378ADD"
RED    = "#E05A5A"
GREEN  = "#1D9E75"
PURPLE = "#9966CC"
BG     = "#0E1117"
CARD   = "#141824"
BORDER = "#2E3450"
TEXT   = "#C0C8D8"
TEXT_H = "#F0F2F6"
PALETTE = [GOLD, BLUE, RED, GREEN, PURPLE, "#CC8844", "#AAAACC"]
 
COLOR_MAP_P = {
    "🏆 Dominants":  GOLD,
    "💪 Solides":    BLUE,
    "⚡ Challengers": GREEN,
    "🌱 Émergents":  RED,
}
 
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=CARD,
    font=dict(color=TEXT, family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT)),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT)),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1, font=dict(color=TEXT)),
    margin=dict(t=50, b=40, l=50, r=30),
)
 
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] {{ font-family:'Inter',sans-serif; background:{BG}; color:{TEXT}; }}
  .stApp {{ background:{BG}; }}
 
  /* ── Top nav bar ── */
  .topbar {{
    display:flex; align-items:center; justify-content:space-between;
    background:linear-gradient(90deg,#0a0d14,#111827);
    border-bottom:1px solid {BORDER};
    padding:10px 32px; margin:-1rem -1rem 24px;
    position:sticky; top:0; z-index:999;
  }}
  .topbar-brand {{
    display:flex; align-items:center; gap:10px;
  }}
  .topbar-brand .logo {{ font-size:1.4rem; }}
  .topbar-brand .title {{ font-size:0.95rem; font-weight:700; color:{TEXT_H}; }}
  .topbar-brand .sub {{ font-size:0.65rem; color:#4A5A7A; letter-spacing:0.08em; margin-top:1px; }}
  .nav-pills {{
    display:flex; gap:6px;
  }}
  .nav-pill {{
    background:none; border:1px solid {BORDER}; border-radius:20px;
    padding:5px 14px; font-size:0.78rem; color:{TEXT}; cursor:pointer;
    font-family:'Inter',sans-serif; transition:all 0.15s;
  }}
  .nav-pill:hover {{ background:rgba(240,192,64,0.08); border-color:{GOLD}; color:{GOLD}; }}
  .nav-pill.active {{
    background:rgba(240,192,64,0.12); border-color:{GOLD};
    color:{GOLD}; font-weight:600;
  }}
 
  /* ── Metric cards ── */
  .metric-card {{
    background:linear-gradient(135deg,{CARD},{CARD});
    border:1px solid {BORDER}; border-radius:12px;
    padding:18px 20px; margin-bottom:8px;
    position:relative; overflow:hidden;
  }}
  .metric-card::before {{
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
  }}
  .metric-card.gold::before  {{ background:linear-gradient(90deg,{GOLD},transparent); }}
  .metric-card.blue::before  {{ background:linear-gradient(90deg,{BLUE},transparent); }}
  .metric-card.green::before {{ background:linear-gradient(90deg,{GREEN},transparent); }}
  .metric-card.red::before   {{ background:linear-gradient(90deg,{RED},transparent); }}
  .metric-card.purple::before{{ background:linear-gradient(90deg,{PURPLE},transparent); }}
  .metric-label {{ font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em; color:#6B7A9A; margin-bottom:5px; }}
  .metric-value {{ font-size:1.9rem; font-weight:700; color:{TEXT_H}; line-height:1; }}
  .metric-sub   {{ font-size:0.72rem; color:#5A6A8A; margin-top:4px; }}
 
  /* ── Section headers ── */
  .section-header {{
    border-left:3px solid {GOLD}; padding-left:12px; margin:28px 0 14px;
  }}
  .section-header h2 {{ color:{TEXT_H}; font-size:1rem; font-weight:600; margin:0; }}
  .section-header p  {{ color:#5A6A8A; font-size:0.76rem; margin:2px 0 0; }}
 
  /* ── Insight box ── */
  .insight-box {{
    background:rgba(240,192,64,0.05); border:1px solid rgba(240,192,64,0.18);
    border-radius:8px; padding:12px 15px; margin:6px 0;
    font-size:0.8rem; color:{TEXT}; line-height:1.55;
  }}
  .insight-box strong {{ color:{GOLD}; }}
 
  /* ── Proba bars ── */
  .proba-wrap {{ margin:10px 0; }}
  .proba-row  {{ display:flex; justify-content:space-between; font-size:0.78rem; margin-bottom:3px; color:{TEXT}; }}
  .proba-bg   {{ height:7px; border-radius:4px; background:rgba(46,52,80,0.8); overflow:hidden; }}
  .proba-fill {{ height:100%; border-radius:4px; }}
 
  /* ── Winner banner ── */
  .winner-banner {{
    background:linear-gradient(135deg,{CARD},#1a1f2e);
    border:1px solid {BORDER}; border-top:2px solid {GOLD};
    border-radius:14px; padding:26px 32px; text-align:center; margin-bottom:20px;
  }}
  .winner-label {{ font-size:0.65rem; text-transform:uppercase; letter-spacing:0.15em; color:#4A5A7A; margin-bottom:10px; }}
  .winner-name  {{ font-size:2rem; font-weight:800; }}
  .winner-meta  {{ font-size:0.75rem; color:#4A5A7A; margin-top:8px; }}
 
  /* ── Tables ── */
  .dataframe {{ font-size:0.78rem !important; }}
 
  /* ── Cluster card ── */
  .cluster-card {{
    background:{CARD}; border:1px solid {BORDER};
    border-radius:10px; padding:14px;
  }}
  .cluster-title {{ font-size:0.85rem; font-weight:600; margin-bottom:10px; }}
  .cluster-team  {{ padding:3px 0; border-bottom:1px solid {BORDER}; font-size:0.76rem; color:{TEXT}; }}
 
  /* Hide streamlit chrome */
  #MainMenu {{ visibility:hidden; }}
  footer {{ visibility:hidden; }}
  [data-testid="stSidebar"] {{ display:none; }}
  .block-container {{ padding-top:0 !important; }}
 
  /* ── Radio stylisé en boutons pill ── */
  div[data-testid="stRadio"] > div {{
    display: flex !important;
    flex-direction: row !important;
    gap: 6px !important;
    flex-wrap: nowrap !important;
  }}
  div[data-testid="stRadio"] label {{
    background: none !important;
    border: 1px solid {BORDER} !important;
    border-radius: 20px !important;
    padding: 5px 14px !important;
    font-size: 0.78rem !important;
    color: {TEXT} !important;
    cursor: pointer !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.15s !important;
    white-space: nowrap !important;
  }}
  div[data-testid="stRadio"] label:hover {{
    background: rgba(240,192,64,0.08) !important;
    border-color: {GOLD} !important;
    color: {GOLD} !important;
  }}
  div[data-testid="stRadio"] label[data-baseweb="radio"] {{
    background: none !important;
  }}
  /* Option sélectionnée */
  div[data-testid="stRadio"] label:has(input:checked) {{
    background: rgba(240,192,64,0.12) !important;
    border-color: {GOLD} !important;
    color: {GOLD} !important;
    font-weight: 600 !important;
  }}
  /* Cacher le cercle radio */
  div[data-testid="stRadio"] input[type="radio"] {{
    display: none !important;
  }}
  div[data-testid="stRadio"] > label {{
    display: none !important;
  }}
</style>
""", unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_all():
    matches, tableau, worldcups = build_dataframes()
    team_stats = build_team_stats(tableau)
    team_stats, k_range, inertias = build_clusters(team_stats)
    clf, sc_m, X_test_s, y_test, features_m = build_ml_model(team_stats, tableau)
    return matches, tableau, worldcups, team_stats, k_range, inertias, clf, sc_m, X_test_s, y_test, features_m
 
with st.spinner("⚽ Chargement…"):
    (matches, tableau, worldcups, team_stats,
     k_range, inertias, clf, sc_m, X_test_s, y_test, features_m) = load_all()
 
ALL_TEAMS   = sorted(team_stats["Team"].tolist())
ALL_YEARS   = sorted(matches["Year"].unique().tolist())
 
FEAT_LABELS = [
    "Win rate (équipe)", "Buts/match (éq.)", "Buts enc./match (éq.)", "Diff. buts (éq.)",
    "Win rate (adv.)",   "Buts/match (adv.)", "Buts enc./match (adv.)", "Diff. buts (adv.)",
]
 
 
PAGES = ["🏠 Vue Générale", "🌍 Équipes", "🤖 Clustering", "🎯 Prédiction", "⚡ Simulateur"]
 
# ══════════════════════════════════════════════════════════════
# TOP NAV — st.radio lié au session_state (navigation stable)
# ══════════════════════════════════════════════════════════════
col_brand, col_nav = st.columns([2, 5])
with col_brand:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:8px 0;">
      <span style="font-size:1.4rem;">⚽</span>
      <div>
        <div style="font-size:0.9rem;font-weight:700;color:{TEXT_H};">WORLD CUP INTELLIGENCE</div>
        <div style="font-size:0.6rem;color:#4A5A7A;letter-spacing:0.08em;">1930–2014 · ML PLATFORM</div>
      </div>
    </div>""", unsafe_allow_html=True)
 
with col_nav:
    page = st.radio(
        "Navigation",
        PAGES,
        horizontal=True,
        label_visibility="collapsed",
        key="page"
    )
 
st.markdown(f'<hr style="border-color:{BORDER};margin:0 0 24px;">', unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def section(title, sub=""):
    st.markdown(f"""
    <div class="section-header">
      <h2>{title}</h2>
      {"<p>"+sub+"</p>" if sub else ""}
    </div>""", unsafe_allow_html=True)
 
def metric_card(label, value, sub="", color="gold"):
    st.markdown(f"""
    <div class="metric-card {color}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)
 
def apply_layout(fig, title="", h=320):
    layout = dict(**PLOTLY_BASE, height=h)
    if title:
        layout["title"] = dict(text=title, font=dict(color=TEXT_H, size=13), x=0.01)
    fig.update_layout(**layout)
    return fig
 
def hex_rgba(hex_color, alpha=0.8):
    r, g, b = int(hex_color[1:3],16), int(hex_color[3:5],16), int(hex_color[5:7],16)
    return f"rgba({r},{g},{b},{alpha})"
 
 
# ══════════════════════════════════════════════════════════════
# PAGE 1 — VUE GÉNÉRALE
# ══════════════════════════════════════════════════════════════
if page == "🏠 Vue Générale":
 
    # ── Filtres ───────────────────────────────────────────────
    f1, f2 = st.columns([2, 2])
    with f1:
        year_range = st.select_slider(
            "Période", options=ALL_YEARS, value=(ALL_YEARS[0], ALL_YEARS[-1]))
    with f2:
        phase_filter = st.multiselect(
            "Phase", ["Groupes","Élimination"], default=["Groupes","Élimination"])
 
    mf = matches[
        (matches["Year"].between(*year_range)) &
        (matches["Phase"].isin(phase_filter if phase_filter else ["Groupes","Élimination"]))
    ].copy()
 
    # ── KPIs ──────────────────────────────────────────────────
    total_goals = int(mf["TotalGoals"].sum())
    avg_goals   = round(mf["TotalGoals"].mean(), 2) if len(mf) else 0
    home_pct    = round((mf["Result"]=="HomeWin").mean()*100,1) if len(mf) else 0
    n_nations   = len(set(mf["HomeTeam"].tolist() + mf["AwayTeam"].tolist()))
    max_row     = mf.loc[mf["TotalGoals"].idxmax()] if len(mf) else None
    record_txt  = (f"{max_row['HomeTeam']} {int(max_row['HomeGoals'])}–"
                   f"{int(max_row['AwayGoals'])} {max_row['AwayTeam']}") if max_row is not None else "—"
 
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: metric_card("Matchs",    f"{len(mf):,}",       f"{mf['Year'].nunique()} éditions",  "gold")
    with c2: metric_card("Buts",      f"{total_goals:,}",   f"moy. {avg_goals}/match",            "blue")
    with c3: metric_card("Dom. wins", f"{home_pct}%",       "avantage domicile",                  "green")
    with c4: metric_card("Nations",   f"{n_nations}",       "pays représentés",                   "purple")
    with c5: metric_card("Record",    record_txt.split()[0] if max_row is not None else "—",
                         record_txt, "red")
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Évolution buts ────────────────────────────────────────
    c1, c2 = st.columns([3,2])
    with c1:
        section("Évolution des buts par édition", "Total (barres) · Moyenne/match (ligne)")
        ge = mf.groupby("Year").agg(Total=("TotalGoals","sum"), N=("Year","count")).reset_index()
        ge["Moy"] = (ge["Total"]/ge["N"]).round(2)
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=ge["Year"], y=ge["Total"], name="Total buts",
            marker_color=GOLD, opacity=0.45, marker_line_color=GOLD, marker_line_width=1),
            secondary_y=False)
        fig.add_trace(go.Scatter(x=ge["Year"], y=ge["Moy"], name="Moy/match",
            mode="lines+markers", line=dict(color=RED,width=2.5), marker=dict(size=7,color=RED)),
            secondary_y=True)
        fig.update_yaxes(title_text="Total buts", color=GOLD, secondary_y=False, gridcolor=BORDER, tickfont=dict(color=GOLD))
        fig.update_yaxes(title_text="Moy/match",  color=RED,  secondary_y=True,  tickfont=dict(color=RED))
        apply_layout(fig, h=340)
        st.plotly_chart(fig, use_container_width=True)
 
    with c2:
        section("Répartition des résultats")
        hw = (mf["Result"]=="HomeWin").sum()
        aw = (mf["Result"]=="AwayWin").sum()
        dr = (mf["Result"]=="Draw").sum()
        fig2 = go.Figure(go.Pie(
            labels=["Victoire domicile","Nul","Victoire extérieur"],
            values=[hw, dr, aw], hole=0.55,
            marker=dict(colors=[BLUE,"#6B7A9A",RED], line=dict(color=BG,width=2)),
            textfont=dict(color=TEXT_H, size=11),
        ))
        apply_layout(fig2, h=340)
        fig2.update_layout(annotations=[dict(
            text=f"<b>{len(mf)}</b><br>matchs",
            font=dict(size=12, color=TEXT_H), showarrow=False)])
        st.plotly_chart(fig2, use_container_width=True)
 
    # ── Distribution + Phase ──────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        section("Distribution des buts par match")
        sf = mf["TotalGoals"].value_counts().sort_index()
        fig3 = go.Figure(go.Bar(
            x=sf.index, y=sf.values,
            marker=dict(color=sf.values,
                        colorscale=[[0,"#2a3050"],[0.5,BLUE],[1,GOLD]],
                        line=dict(color=BG,width=0.8)),
            text=sf.values, textposition="outside", textfont=dict(color=TEXT,size=10)
        ))
        apply_layout(fig3, h=300)
        fig3.update_layout(xaxis_title="Buts", yaxis_title="Nb matchs")
        st.plotly_chart(fig3, use_container_width=True)
 
    with c4:
        section("Groupes vs Élimination directe")
        fig4 = go.Figure()
        for ph, color in [("Groupes",GOLD),("Élimination",RED)]:
            data = mf[mf["Phase"]==ph]["TotalGoals"]
            if len(data):
                freq = data.value_counts().sort_index()
                fig4.add_trace(go.Bar(x=freq.index, y=freq.values,
                    name=f"{ph} (moy={data.mean():.2f})",
                    marker_color=color, opacity=0.55,
                    marker_line_color=BG, marker_line_width=0.8))
        fig4.update_layout(barmode="group", **PLOTLY_BASE, height=300)
        fig4.update_layout(xaxis_title="Buts", yaxis_title="Fréquence")
        st.plotly_chart(fig4, use_container_width=True)
 
    # ── Affluence ─────────────────────────────────────────────
    section("Affluence & expansion du tournoi")
    wc_f = worldcups[worldcups["Year"].between(*year_range)]
    fig5 = make_subplots(specs=[[{"secondary_y":True}]])
    fig5.add_trace(go.Scatter(x=wc_f["Year"], y=wc_f["Attendance"],
        name="Affluence totale", mode="lines+markers",
        fill="tozeroy", fillcolor="rgba(55,138,221,0.1)",
        line=dict(color=BLUE,width=2.5), marker=dict(size=7)),
        secondary_y=False)
    fig5.add_trace(go.Scatter(x=wc_f["Year"], y=wc_f["QualifiedTeams"],
        name="Équipes qualifiées", mode="lines+markers",
        line=dict(color=GOLD,width=2,dash="dash"), marker=dict(size=7,symbol="diamond")),
        secondary_y=True)
    fig5.update_yaxes(title_text="Affluence", color=BLUE, secondary_y=False,
                      gridcolor=BORDER, tickfont=dict(color=BLUE))
    fig5.update_yaxes(title_text="Équipes",   color=GOLD, secondary_y=True,
                      tickfont=dict(color=GOLD))
    apply_layout(fig5, h=280)
    st.plotly_chart(fig5, use_container_width=True)
 
    # ── Insights ──────────────────────────────────────────────
    section("Insights Clés")
    i1,i2,i3 = st.columns(3)
    with i1: st.markdown("""<div class="insight-box"><strong>📉 Déclin offensif</strong><br/>
        Moyenne de buts : <strong>5.38/match</strong> en 1954 → <strong>2.21</strong> en 1990.
        Le jeu s'est densifié tactiquement sur 60 ans.</div>""", unsafe_allow_html=True)
    with i2: st.markdown("""<div class="insight-box"><strong>🏠 Avantage domicile</strong><br/>
        57% des matchs gagnés par l'équipe "domicile" (conventionnel) vs 20% pour l'extérieur.
        L'effet psychologique est réel.</div>""", unsafe_allow_html=True)
    with i3: st.markdown("""<div class="insight-box"><strong>⚡ Phase éliminatoire</strong><br/>
        Les matchs à élimination directe produisent <strong>moins de buts</strong> —
        pression défensive et enjeux plus élevés.</div>""", unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE PAR ÉQUIPE
# ══════════════════════════════════════════════════════════════
elif page == "🌍 Équipes":
 
    # ── Filtre année (spécifique à cette page) ────────────────
    col_f1, col_f2, col_f3 = st.columns([2,2,2])
    with col_f1:
        year_sel = st.selectbox("Filtrer par édition", ["Toutes les éditions"] + [str(y) for y in ALL_YEARS])
    with col_f2:
        phase_eq = st.multiselect("Phase", ["Groupes","Élimination"],
                                  default=["Groupes","Élimination"], key="phase_eq")
 
    # Construire team_stats filtrée si nécessaire
    if year_sel != "Toutes les éditions":
        tab_f = tableau[tableau["Year"] == int(year_sel)].copy()
        tab_f["Result"] = tab_f.apply(
            lambda r: "Win" if r["Team G"]>r["Opponent G"]
            else ("Loss" if r["Team G"]<r["Opponent G"] else "Draw"), axis=1)
        ts_page = build_team_stats(tab_f)
        ts_page["Profil"] = ts_page["Team"].map(
            team_stats.set_index("Team")["Profil"]).fillna("⚡ Challengers")
        subtitle = f"Édition {year_sel}"
    else:
        ts_page = team_stats.copy()
        subtitle = "1930–2014 (toutes éditions)"
 
    matches_eq = matches[matches["Phase"].isin(phase_eq if phase_eq else ["Groupes","Élimination"])]
    if year_sel != "Toutes les éditions":
        matches_eq = matches_eq[matches_eq["Year"] == int(year_sel)]
 
    st.markdown(f"<p style='color:#5A6A8A;font-size:0.8rem;margin-bottom:16px;'>{subtitle} · {len(ts_page)} nations</p>",
                unsafe_allow_html=True)
 
    # ── Top 15 ───────────────────────────────────────────────
    cl, cr = st.columns(2)
    with cl:
        section("Top 15 — Victoires")
        top15 = ts_page.nlargest(15,"Victoires").reset_index(drop=True)
        fig = go.Figure(go.Bar(
            y=top15["Team"][::-1], x=top15["Victoires"][::-1], orientation="h",
            marker=dict(color=top15["Victoires"][::-1].tolist(),
                        colorscale=[[0,"#2a3050"],[0.5,BLUE],[1,GOLD]],
                        line=dict(color=BG,width=0.5)),
            text=top15["Victoires"][::-1], textposition="outside",
            textfont=dict(color=TEXT_H,size=10)
        ))
        apply_layout(fig, h=420)
        fig.update_layout(xaxis_title="Victoires", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
 
    with cr:
        section("Top 15 — Buts marqués")
        top15b = ts_page.nlargest(15,"Buts_marques").reset_index(drop=True)
        fig2 = go.Figure(go.Bar(
            y=top15b["Team"][::-1], x=top15b["Buts_marques"][::-1], orientation="h",
            marker=dict(color=top15b["Buts_marques"][::-1].tolist(),
                        colorscale=[[0,"#1a2a3a"],[0.5,GREEN],[1,GOLD]],
                        line=dict(color=BG,width=0.5)),
            text=top15b["Buts_marques"][::-1], textposition="outside",
            textfont=dict(color=TEXT_H,size=10)
        ))
        apply_layout(fig2, h=420)
        fig2.update_layout(xaxis_title="Buts marqués", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)
 
    # ── Scatter ───────────────────────────────────────────────
    section("Profil offensif vs Win rate", "Taille = matchs joués · Couleur = profil ML")
    ts_sc = ts_page[ts_page["Matchs"] >= 2].copy()
    fig3 = go.Figure()
    for profil in PROFIL_ORDER:
        sub = ts_sc[ts_sc["Profil"]==profil]
        if sub.empty: continue
        c = COLOR_MAP_P.get(profil, PURPLE)
        fig3.add_trace(go.Scatter(
            x=sub["Buts_pm"], y=sub["Win_rate"], mode="markers+text",
            name=profil,
            marker=dict(size=np.clip(np.sqrt(sub["Matchs"])*4,6,30),
                        color=hex_rgba(c,0.8), line=dict(color=BG,width=1)),
            text=sub.apply(lambda r: r["Team"] if r["Matchs"]>=15 else "", axis=1),
            textposition="top center", textfont=dict(size=8,color=TEXT),
            hovertemplate="<b>%{customdata[0]}</b><br>Win rate:%{y:.1f}%<br>Buts/m:%{x:.2f}<br>Matchs:%{customdata[1]}<extra></extra>",
            customdata=sub[["Team","Matchs"]].values
        ))
    fig3.add_hline(y=ts_sc["Win_rate"].mean(), line_dash="dash", line_color="#444466", line_width=1, opacity=0.5)
    fig3.add_vline(x=ts_sc["Buts_pm"].mean(),  line_dash="dash", line_color="#444466", line_width=1, opacity=0.5)
    apply_layout(fig3, h=460)
    fig3.update_layout(xaxis_title="Buts/match", yaxis_title="Win rate (%)")
    st.plotly_chart(fig3, use_container_width=True)
 
    # ── Radar ─────────────────────────────────────────────────
    section("Radar — Comparaison multi-équipes", "Sélectionnez jusqu'à 6 équipes")
    default_r = [t for t in ["Brazil","Germany","Italy","Argentina","France"] if t in ts_page["Team"].tolist()]
    sel_teams = st.multiselect("Équipes", ts_page["Team"].sort_values().tolist(),
                               default=default_r[:5], max_selections=6, key="radar_sel")
    if len(sel_teams) >= 2:
        ts_r = ts_page.copy()
        # Normalisation robuste
        for col, alias, inv in [
            ("Win_rate","n_wr",False), ("Buts_pm","n_bpm",False),
            ("Buts_encais_pm","n_def",True), ("Matchs","n_lon",False), ("Diff_buts","n_eff",False)
        ]:
            rng = ts_r[col].max() - ts_r[col].min() + 1e-9
            ts_r[alias] = (ts_r[col] - ts_r[col].min()) / rng
            if inv: ts_r[alias] = 1 - ts_r[alias]
 
        categories = ["Win rate","Buts/match","Solidité déf.","Longévité","Efficacité"]
        fig4 = go.Figure()
        for team, color in zip(sel_teams, PALETTE):
            row = ts_r[ts_r["Team"]==team]
            if row.empty: continue
            row = row.iloc[0]
            vals = [row["n_wr"], row["n_bpm"], row["n_def"], row["n_lon"], row["n_eff"]]
            vals_c = vals + [vals[0]]
            cats_c = categories + [categories[0]]
            r_c,g_c,b_c = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
            fig4.add_trace(go.Scatterpolar(
                r=vals_c, theta=cats_c, fill="toself", name=team,
                line=dict(color=color, width=2),
                fillcolor=f"rgba({r_c},{g_c},{b_c},0.12)",
            ))
        fig4.update_layout(
            polar=dict(
                bgcolor=CARD,
                radialaxis=dict(visible=True, range=[0,1], tickfont=dict(color="#5A6A8A"),
                                gridcolor=BORDER, color=TEXT),
                angularaxis=dict(color=TEXT, gridcolor=BORDER)
            ),
            **PLOTLY_BASE, height=440
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Sélectionnez au moins 2 équipes pour afficher le radar.")
 
    # ── Fiche Nation ──────────────────────────────────────────
    section("Fiche Nation", "Statistiques détaillées + historique par édition")
    avail_teams = ts_page["Team"].sort_values().tolist()
    default_idx = avail_teams.index("Brazil") if "Brazil" in avail_teams else 0
    team_sel = st.selectbox("Choisir une nation", avail_teams, index=default_idx, key="fiche_sel")
 
    row = ts_page[ts_page["Team"]==team_sel].iloc[0]
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: metric_card("Matchs",    int(row["Matchs"]),   "", "gold")
    with c2: metric_card("Victoires", int(row["Victoires"]), f"Win rate {row['Win_rate']}%", "green")
    with c3: metric_card("Buts +",    int(row["Buts_marques"]), f"{row['Buts_pm']}/match", "blue")
    with c4: metric_card("Buts -",    int(row["Buts_encais"]),  f"{row['Buts_encais_pm']}/match", "red")
    with c5: metric_card("Diff",      f"{int(row['Diff_buts']):+d}", f"{int(row['Editions'])} éditions", "purple")
 
    # Historique par édition (sur toutes les années, indépendamment du filtre)
    nation_hist = matches[(matches["HomeTeam"]==team_sel)|(matches["AwayTeam"]==team_sel)].copy()
    if len(nation_hist) > 0:
        scored, conceded, years_n = [], [], []
        for yr, grp in nation_hist.groupby("Year"):
            s = grp.apply(lambda r: r["HomeGoals"] if r["HomeTeam"]==team_sel else r["AwayGoals"], axis=1).sum()
            c = grp.apply(lambda r: r["AwayGoals"] if r["HomeTeam"]==team_sel else r["HomeGoals"], axis=1).sum()
            scored.append(int(s)); conceded.append(int(c)); years_n.append(yr)
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=years_n, y=scored,   name="Buts marqués",   marker_color=GREEN, opacity=0.85))
        fig5.add_trace(go.Bar(x=years_n, y=[-c for c in conceded], name="Buts encaissés", marker_color=RED, opacity=0.85))
        apply_layout(fig5, f"Bilan de {team_sel} par édition", h=260)
        fig5.update_layout(barmode="relative", yaxis_title="Buts (+ marqués / - encaissés)")
        st.plotly_chart(fig5, use_container_width=True)
 
 
# ══════════════════════════════════════════════════════════════
# PAGE 3 — CLUSTERING ML
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Clustering":
    st.markdown(f"""
    <h1 style="color:{TEXT_H};font-size:1.5rem;font-weight:700;margin-bottom:4px;">
      Clustering ML <span style="color:{GOLD};">·</span> K-Means — Profils de Jeu
    </h1>
    <p style="color:#4A5A7A;font-size:0.8rem;margin-bottom:24px;">
      Segmentation non supervisée · 5 features · k=4 clusters optimaux · 79 nations
    </p>""", unsafe_allow_html=True)
 
    # ── Méthode du coude + Répartition ───────────────────────
    cl, cr = st.columns([3,2])
    with cl:
        section("Méthode du coude", "Validation du k optimal")
        fig = go.Figure(go.Scatter(x=list(k_range), y=inertias, mode="lines+markers",
            line=dict(color=GOLD,width=2.5), marker=dict(size=9,color=GOLD,line=dict(color=BG,width=1.5))))
        idx4 = list(k_range).index(4)
        fig.add_vline(x=4, line_dash="dash", line_color=RED, line_width=1.5, opacity=0.7)
        fig.add_annotation(x=4.15, y=inertias[idx4], text="k=4 optimal",
                           font=dict(color=RED,size=11), showarrow=False, xanchor="left")
        apply_layout(fig, h=300)
        fig.update_layout(xaxis=dict(tickmode="linear",tick0=2,dtick=1,gridcolor=BORDER),
                          xaxis_title="Clusters (k)", yaxis_title="Inertie WCSS")
        st.plotly_chart(fig, use_container_width=True)
 
    with cr:
        section("Répartition des profils")
        pc = team_stats["Profil"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=pc.index, values=pc.values, hole=0.5,
            marker=dict(colors=[COLOR_MAP_P.get(p,PURPLE) for p in pc.index],
                        line=dict(color=BG,width=2)),
            textfont=dict(color=TEXT_H,size=10)
        ))
        apply_layout(fig2, h=300)
        fig2.update_layout(annotations=[dict(
            text=f"<b>{len(team_stats)}</b><br>nations",
            font=dict(size=12,color=TEXT_H), showarrow=False)])
        st.plotly_chart(fig2, use_container_width=True)
 
    # ── Scatter double ────────────────────────────────────────
    cl2, cr2 = st.columns(2)
    with cl2:
        section("Win rate vs Offensivité")
        fig3 = go.Figure()
        for profil in PROFIL_ORDER:
            sub = team_stats[team_stats["Profil"]==profil]
            if sub.empty: continue
            c = COLOR_MAP_P.get(profil,PURPLE)
            fig3.add_trace(go.Scatter(
                x=sub["Buts_pm"], y=sub["Win_rate"], mode="markers",
                name=profil,
                marker=dict(size=np.clip(np.sqrt(sub["Matchs"])*3.5,6,24),
                            color=hex_rgba(c,0.8), line=dict(color=BG,width=0.5)),
                hovertemplate="<b>%{customdata}</b><br>Win:%{y:.1f}% Buts:%{x:.2f}<extra></extra>",
                customdata=sub["Team"].values
            ))
        apply_layout(fig3, h=360)
        fig3.update_layout(xaxis_title="Buts/match", yaxis_title="Win rate (%)")
        st.plotly_chart(fig3, use_container_width=True)
 
    with cr2:
        section("Attaque vs Défense")
        fig4 = go.Figure()
        for profil in PROFIL_ORDER:
            sub = team_stats[team_stats["Profil"]==profil]
            if sub.empty: continue
            c = COLOR_MAP_P.get(profil,PURPLE)
            fig4.add_trace(go.Scatter(
                x=sub["Buts_encais_pm"], y=sub["Buts_pm"], mode="markers",
                name=profil,
                marker=dict(size=np.clip(np.sqrt(sub["Matchs"])*3.5,6,24),
                            color=hex_rgba(c,0.8), line=dict(color=BG,width=0.5)),
                hovertemplate="<b>%{customdata}</b><br>Enc:%{x:.2f} Mar:%{y:.2f}<extra></extra>",
                customdata=sub["Team"].values
            ))
        mv = max(team_stats["Buts_encais_pm"].max(), team_stats["Buts_pm"].max())
        fig4.add_trace(go.Scatter(x=[0,mv],y=[0,mv],mode="lines",
            line=dict(color="#444466",dash="dash",width=1),name="Équilibre"))
        apply_layout(fig4, h=360)
        fig4.update_layout(xaxis_title="Buts encaissés/match", yaxis_title="Buts marqués/match")
        st.plotly_chart(fig4, use_container_width=True)
 
    # ── Équipes par profil ────────────────────────────────────
    section("Équipes par profil de jeu")
    p_cols = st.columns(4)
    for col, profil in zip(p_cols, PROFIL_ORDER):
        with col:
            sub = sorted(team_stats[team_stats["Profil"]==profil]["Team"].tolist())
            c = COLOR_MAP_P.get(profil,PURPLE)
            r_c,g_c,b_c = int(c[1:3],16),int(c[3:5],16),int(c[5:7],16)
            teams_html = "".join([
                f'<div class="cluster-team">{t}</div>' for t in sub])
            st.markdown(f"""
            <div class="cluster-card" style="border-top:2px solid {c};">
              <div class="cluster-title" style="color:{c};">
                {profil}
                <span style="background:rgba({r_c},{g_c},{b_c},0.15);color:{c};
                      padding:2px 8px;border-radius:10px;font-size:0.68rem;margin-left:6px;">
                  {len(sub)} nations
                </span>
              </div>
              {teams_html}
            </div>""", unsafe_allow_html=True)
 
    # ── Stats moyennes ────────────────────────────────────────
    section("Statistiques moyennes par profil")
    cm_df = (team_stats.groupby("Profil")[["Win_rate","Buts_pm","Buts_encais_pm","Diff_buts","Matchs"]]
             .mean().round(2).reindex(PROFIL_ORDER))
    cm_df.columns = ["Win rate (%)","Buts/match","Buts enc./match","Diff. buts","Matchs moy."]
    # Affichage sans background_gradient (bug Streamlit/pandas) — on formate directement
    st.dataframe(cm_df.style.format("{:.2f}").highlight_max(axis=0, color="rgba(240,192,64,0.25)")
                             .highlight_min(axis=0, color="rgba(224,90,90,0.15)"),
                 use_container_width=True)
 
 
# ══════════════════════════════════════════════════════════════
# PAGE 4 — PRÉDICTION ML
# ══════════════════════════════════════════════════════════════
elif page == "🎯 Prédiction":
    st.markdown(f"""
    <h1 style="color:{TEXT_H};font-size:1.5rem;font-weight:700;margin-bottom:4px;">
      Prédiction ML <span style="color:{GOLD};">·</span> Régression Logistique
    </h1>
    <p style="color:#4A5A7A;font-size:0.8rem;margin-bottom:24px;">
      Modèle entraîné sur 80% des données · évalué sur 20% · 8 features
    </p>""", unsafe_allow_html=True)
 
    y_pred = clf.predict(X_test_s)
    acc    = round(clf.score(X_test_s, y_test)*100, 1)
 
    # ── KPIs modèle ───────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Accuracy", f"{acc}%", "Régression Logistique", "gold")
    with c2: metric_card("Train / Test", "80 / 20%", "split stratifié", "blue")
    with c3: metric_card("Features", "8", "statistiques historiques", "green")
    with c4: metric_card("Classes", "3", "Victoire · Nul · Défaite", "purple")
 
    # ── Matrice de confusion ──────────────────────────────────
    cl, cr = st.columns(2)
    with cl:
        section("Matrice de confusion", f"Accuracy globale : {acc}%")
        cm = confusion_matrix(y_test, y_pred)
        labels_cm = ["Défaite","Nul","Victoire"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels_cm, y=labels_cm,
            colorscale=[[0,"#1a1f2e"],[0.5,BLUE],[1,GOLD]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=16,color=TEXT_H), showscale=True
        ))
        apply_layout(fig, h=320)
        fig.update_layout(
            xaxis_title="Prédit", yaxis_title="Réel",
            yaxis=dict(autorange="reversed", gridcolor=BORDER),
            xaxis=dict(gridcolor=BORDER)
        )
        st.plotly_chart(fig, use_container_width=True)
 
    # ── Feature importance ────────────────────────────────────
    with cr:
        section("Importance des features", "Coefficients absolus moyens")
        coef_mean = np.abs(clf.coef_).mean(axis=0)
        idx = np.argsort(coef_mean)
        bar_colors = [GOLD if i >= 4 else BLUE for i in idx]
        fig2 = go.Figure(go.Bar(
            y=[FEAT_LABELS[i] for i in idx], x=coef_mean[idx],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color=BG,width=0.5)),
            text=[f"{v:.3f}" for v in coef_mean[idx]],
            textposition="outside", textfont=dict(color=TEXT,size=9)
        ))
        # Légende manuelle
        fig2.add_trace(go.Bar(x=[0],y=[""],marker_color=GOLD,name="Adversaire",orientation="h",showlegend=True))
        fig2.add_trace(go.Bar(x=[0],y=[""],marker_color=BLUE,name="Équipe",orientation="h",showlegend=True))
        apply_layout(fig2, h=320)
        fig2.update_layout(xaxis_title="Importance absolue")
        st.plotly_chart(fig2, use_container_width=True)
 
    # ── Rapport de classification ─────────────────────────────
    section("Rapport de classification détaillé")
    report = classification_report(
        y_test, y_pred,
        target_names=["Défaite","Nul","Victoire"],
        output_dict=True
    )
    # Construire un DataFrame propre sans la ligne "support" problématique
    report_rows = {k: v for k, v in report.items() if isinstance(v, dict)}
    report_df = pd.DataFrame(report_rows).T.round(3)
    report_df = report_df[["precision","recall","f1-score","support"]].copy()
    report_df["support"] = report_df["support"].astype(int)
    st.dataframe(
        report_df.style.format({"precision":"{:.3f}","recall":"{:.3f}","f1-score":"{:.3f}","support":"{:d}"})
                 .highlight_max(axis=0, subset=["precision","recall","f1-score"],
                                color="rgba(240,192,64,0.2)")
                 .highlight_min(axis=0, subset=["precision","recall","f1-score"],
                                color="rgba(224,90,90,0.1)"),
        use_container_width=True
    )
 
    # ── Insights ──────────────────────────────────────────────
    section("Interprétation du modèle")
    i1,i2,i3 = st.columns(3)
    with i1: st.markdown(f"""<div class="insight-box"><strong>🎯 Accuracy {acc}%</strong><br/>
        Cohérent pour un sport imprévisible. Les meilleurs modèles sport peinent à dépasser 62%
        sans données individuelles.</div>""", unsafe_allow_html=True)
    with i2: st.markdown("""<div class="insight-box"><strong>🔑 Features clés</strong><br/>
        Le <strong>win rate historique</strong> et la <strong>différence de buts</strong>
        sont les prédicteurs dominants dans les deux équipes.</div>""", unsafe_allow_html=True)
    with i3: st.markdown("""<div class="insight-box"><strong>⚠️ Limites</strong><br/>
        Pas de données individuelles (effectifs, blessés, forme récente) ni de contexte
        situationnel (stade, enjeu de la rencontre).</div>""", unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════
# PAGE 5 — SIMULATEUR
# ══════════════════════════════════════════════════════════════
elif page == "⚡ Simulateur":
    st.markdown(f"""
    <h1 style="color:{TEXT_H};font-size:1.5rem;font-weight:700;margin-bottom:4px;">
      Simulateur <span style="color:{GOLD};">·</span> Prédire un Match
    </h1>
    <p style="color:#4A5A7A;font-size:0.8rem;margin-bottom:24px;">
      Pronostic basé sur les statistiques historiques FIFA 1930–2014
    </p>""", unsafe_allow_html=True)
 
    # ── Sélection équipes ────────────────────────────────────
    ca, cv, cb = st.columns([5,1,5])
    with ca:
        team_a = st.selectbox("Équipe A", ALL_TEAMS,
                              index=ALL_TEAMS.index("Brazil") if "Brazil" in ALL_TEAMS else 0,
                              key="sim_a")
    with cv:
        st.markdown(f"""<div style="text-align:center;padding-top:30px;font-size:1.4rem;
                        font-weight:700;color:{GOLD};">VS</div>""", unsafe_allow_html=True)
    with cb:
        team_b = st.selectbox("Équipe B", ALL_TEAMS,
                              index=ALL_TEAMS.index("Germany") if "Germany" in ALL_TEAMS else 1,
                              key="sim_b")
 
    if team_a == team_b:
        st.warning("⚠️ Veuillez sélectionner deux équipes différentes.")
    else:
        res = predict_match(team_a, team_b, team_stats, clf, sc_m)
        if res:
            p_a, p_d, p_b = res["p_win_a"], res["p_draw"], res["p_win_b"]
            probs = [p_a, p_d, p_b]
            winner_idx = probs.index(max(probs))
            winner = [team_a, "Match nul", team_b][winner_idx]
            winner_color = [GREEN, "#6B7A9A", RED][winner_idx]
 
            # ── Bannière résultat ─────────────────────────────
            emoji = "🏆" if winner_idx != 1 else "🤝"
            st.markdown(f"""
            <div class="winner-banner">
              <div class="winner-label">Pronostic · Régression Logistique</div>
              <div class="winner-name" style="color:{winner_color};">{emoji} {winner}</div>
              <div class="winner-meta">Basé sur les statistiques historiques FIFA 1930–2014</div>
            </div>""", unsafe_allow_html=True)
 
            # ── Probabilités + Bar chart ──────────────────────
            cp1, cp2 = st.columns([2,3])
            with cp1:
                section("Probabilités")
                for label, prob, color in [
                    (f"Victoire {team_a}", p_a, GREEN),
                    ("Match nul",          p_d, "#6B7A9A"),
                    (f"Victoire {team_b}", p_b, RED),
                ]:
                    pct = prob*100
                    st.markdown(f"""
                    <div class="proba-wrap">
                      <div class="proba-row">
                        <span>{label}</span>
                        <span style="color:{color};font-weight:600;">{pct:.1f}%</span>
                      </div>
                      <div class="proba-bg">
                        <div class="proba-fill" style="width:{pct}%;background:{color};"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
 
            with cp2:
                fig = go.Figure(go.Bar(
                    x=[f"Victoire\n{team_a}", "Nul", f"Victoire\n{team_b}"],
                    y=probs,
                    marker=dict(
                        color=[GREEN, "#6B7A9A", RED],
                        line=dict(color=BG,width=1)
                    ),
                    text=[f"{v*100:.1f}%" for v in probs],
                    textposition="outside", textfont=dict(size=13,color=TEXT_H)
                ))
                apply_layout(fig, h=260)
                fig.update_layout(
                    yaxis=dict(tickformat=".0%",range=[0,max(probs)*1.35],gridcolor=BORDER),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
 
            # ── Comparaison profils ───────────────────────────
            section("Comparaison des profils")
            ra = team_stats[team_stats["Team"]==team_a].iloc[0]
            rb = team_stats[team_stats["Team"]==team_b].iloc[0]
 
            # Radar comparatif
            ts_r = team_stats.copy()
            for col, alias, inv in [
                ("Win_rate","n_wr",False),("Buts_pm","n_bpm",False),
                ("Buts_encais_pm","n_def",True),("Matchs","n_lon",False),("Diff_buts","n_eff",False)
            ]:
                rng = ts_r[col].max()-ts_r[col].min()+1e-9
                ts_r[alias] = (ts_r[col]-ts_r[col].min())/rng
                if inv: ts_r[alias] = 1-ts_r[alias]
 
            cats = ["Win rate","Buts/match","Solidité déf.","Longévité","Efficacité"]
            fig_r = go.Figure()
            for team, color in [(team_a, GOLD), (team_b, BLUE)]:
                row = ts_r[ts_r["Team"]==team].iloc[0]
                vals = [row["n_wr"],row["n_bpm"],row["n_def"],row["n_lon"],row["n_eff"]]
                vals += [vals[0]]; c_cats = cats+[cats[0]]
                r_c,g_c,b_c = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
                fig_r.add_trace(go.Scatterpolar(
                    r=vals, theta=c_cats, fill="toself", name=team,
                    line=dict(color=color,width=2),
                    fillcolor=f"rgba({r_c},{g_c},{b_c},0.12)"
                ))
            fig_r.update_layout(
                polar=dict(bgcolor=CARD,
                    radialaxis=dict(visible=True,range=[0,1],tickfont=dict(color="#5A6A8A"),
                                   gridcolor=BORDER,color=TEXT),
                    angularaxis=dict(color=TEXT,gridcolor=BORDER)),
                **PLOTLY_BASE, height=380
            )
 
            col_r, col_t = st.columns([3,2])
            with col_r:
                st.plotly_chart(fig_r, use_container_width=True)
            with col_t:
                comp = {
                    "Indicateur":["Win rate","Buts/match","Buts enc./match","Diff. buts","Matchs","Victoires"],
                    team_a: [f"{ra['Win_rate']}%", ra['Buts_pm'], ra['Buts_encais_pm'],
                             f"{int(ra['Diff_buts']):+d}", int(ra['Matchs']), int(ra['Victoires'])],
                    team_b: [f"{rb['Win_rate']}%", rb['Buts_pm'], rb['Buts_encais_pm'],
                             f"{int(rb['Diff_buts']):+d}", int(rb['Matchs']), int(rb['Victoires'])],
                }
                st.dataframe(pd.DataFrame(comp).set_index("Indicateur"), use_container_width=True)
 
    # ── Quarts de finale imaginaires ──────────────────────────
    section("Simulation — Quarts de finale imaginaires",
            "Matchups historiques scénarisés · basé sur les stats 1930–2014")
    matchups_qf = [
        ("Brazil","Germany"), ("Argentina","Italy"),
        ("France","Spain"),   ("Netherlands","England"),
    ]
    available_qf = [(a,b) for a,b in matchups_qf if a in ALL_TEAMS and b in ALL_TEAMS]
    if available_qf:
        qf_cols = st.columns(len(available_qf))
        for col, (ta, tb) in zip(qf_cols, available_qf):
            with col:
                res_qf = predict_match(ta, tb, team_stats, clf, sc_m)
                if not res_qf: continue
                pa, pd_, pb = res_qf["p_win_a"], res_qf["p_draw"], res_qf["p_win_b"]
                wi = [pa, pd_, pb].index(max(pa, pd_, pb))
                bar_colors_qf = [GREEN if i==wi else ("#6B7A9A" if i==1 else BLUE) for i in range(3)]
                fig_qf = go.Figure(go.Bar(
                    x=[ta, "Nul", tb], y=[pa, pd_, pb],
                    marker=dict(color=bar_colors_qf, line=dict(color=BG,width=0.8)),
                    text=[f"{v*100:.1f}%" for v in [pa, pd_, pb]],
                    textposition="outside", textfont=dict(size=10,color=TEXT_H)
                ))
                fig_qf.update_layout(
                    title=dict(text=f"<b>{ta}</b> vs <b>{tb}</b>",
                               font=dict(color=GOLD,size=11),x=0.5),
                    yaxis=dict(tickformat=".0%",range=[0,0.85],gridcolor=BORDER),
                    **PLOTLY_BASE, height=260,
                    margin=dict(t=45,b=30,l=20,r=20), showlegend=False
                )
                st.plotly_chart(fig_qf, use_container_width=True)
 
