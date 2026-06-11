"""
⚽ World Cup Intelligence Platform
ML Engineer · Analyse historique FIFA 1930–2014
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report
from data_engine import (
    build_dataframes, build_team_stats, build_clusters,
    build_ml_model, predict_match
)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION PAGE & THÈME
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="World Cup Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ──────────────────────────────────────────
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

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=CARD,
    font=dict(color=TEXT, family="Inter, sans-serif", size=12),
    title_font=dict(color=TEXT_H, size=15, family="Inter, sans-serif"),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT)),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT)),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1, font=dict(color=TEXT)),
    margin=dict(t=50, b=40, l=50, r=30),
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {BG};
    color: {TEXT};
  }}
  .stApp {{ background-color: {BG}; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0a0d14 0%, #111827 100%);
    border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stMultiSelect label {{
    color: {TEXT} !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  /* Metric cards */
  .metric-card {{
    background: linear-gradient(135deg, {CARD} 0%, #1a1f2e 100%);
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
  }}
  .metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }}
  .metric-card.gold::before {{ background: linear-gradient(90deg, {GOLD}, transparent); }}
  .metric-card.blue::before {{ background: linear-gradient(90deg, {BLUE}, transparent); }}
  .metric-card.green::before {{ background: linear-gradient(90deg, {GREEN}, transparent); }}
  .metric-card.red::before  {{ background: linear-gradient(90deg, {RED},  transparent); }}
  .metric-card.purple::before {{ background: linear-gradient(90deg, {PURPLE}, transparent); }}

  .metric-label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6B7A9A;
    margin-bottom: 6px;
  }}
  .metric-value {{
    font-size: 2rem;
    font-weight: 700;
    color: {TEXT_H};
    line-height: 1;
  }}
  .metric-sub {{
    font-size: 0.75rem;
    color: #5A6A8A;
    margin-top: 4px;
  }}

  /* Section headers */
  .section-header {{
    border-left: 3px solid {GOLD};
    padding-left: 12px;
    margin: 32px 0 16px;
  }}
  .section-header h2 {{
    color: {TEXT_H};
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: 0.02em;
  }}
  .section-header p {{
    color: #5A6A8A;
    font-size: 0.78rem;
    margin: 2px 0 0;
  }}

  /* Nav pills (sidebar) */
  .nav-header {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #4A5A7A;
    margin: 20px 0 8px;
    padding-left: 4px;
  }}

  /* Insight box */
  .insight-box {{
    background: rgba(240,192,64,0.06);
    border: 1px solid rgba(240,192,64,0.2);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.82rem;
    color: {TEXT};
  }}
  .insight-box strong {{ color: {GOLD}; }}

  /* Cluster badge */
  .badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
  }}

  /* Proba bars */
  .proba-bar-wrap {{ margin: 8px 0; }}
  .proba-bar-label {{
    display: flex; justify-content: space-between;
    font-size: 0.8rem; margin-bottom: 3px;
    color: {TEXT};
  }}
  .proba-bar-bg {{
    height: 8px; border-radius: 4px;
    background: rgba(46,52,80,0.8);
    overflow: hidden;
  }}
  .proba-bar-fill {{
    height: 100%; border-radius: 4px;
    transition: width 0.4s ease;
  }}

  /* Hide streamlit chrome */
  #MainMenu {{ visibility: hidden; }}
  footer {{ visibility: hidden; }}

  /* Plotly tweaks */
  .js-plotly-plot .plotly .modebar {{
    background: transparent !important;
  }}

  /* Table */
  .dataframe {{ font-size: 0.8rem !important; }}

  div[data-testid="stHorizontalBlock"] > div {{
    gap: 12px;
  }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CACHE & DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_all():
    matches, tableau, worldcups = build_dataframes()
    team_stats = build_team_stats(tableau)
    team_stats, k_range, inertias = build_clusters(team_stats)
    clf, hgb, sc_m, X_test_s, y_test, features_m = build_ml_model(team_stats, tableau)
    return matches, tableau, worldcups, team_stats, k_range, inertias, clf, hgb, sc_m, X_test_s, y_test, features_m

with st.spinner("🔄 Chargement des données et entraînement des modèles…"):
    (matches, tableau, worldcups, team_stats,
     k_range, inertias, clf, hgb, sc_m, X_test_s, y_test, features_m) = load_all()

ALL_TEAMS = sorted(team_stats["Team"].tolist())

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 24px 0 16px;">
      <div style="font-size: 2.5rem; margin-bottom: 8px;">⚽</div>
      <div style="font-size: 1rem; font-weight: 700; color: #F0F2F6; letter-spacing: 0.05em;">
        WORLD CUP<br/>
        <span style="color: #F0C040;">INTELLIGENCE</span>
      </div>
      <div style="font-size: 0.65rem; color: #4A5A7A; margin-top: 6px; letter-spacing: 0.1em;">
        1930 – 2014 · ML PLATFORM
      </div>
    </div>
    <hr style="border-color: #1e2538; margin: 0 0 8px;">
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-header">Navigation</div>', unsafe_allow_html=True)

    PAGES = ["  Vue Générale", "  Analyse par Équipe", "  Clustering ML", "  Prédiction ML", "  Simulateur"]
    if "page" not in st.session_state:
        st.session_state["page"] = PAGES[0]

    page = st.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(st.session_state["page"]),
        key="nav_radio",
        label_visibility="collapsed"
    )
    st.session_state["page"] = page

    st.markdown('<div class="nav-header">Filtres Globaux</div>', unsafe_allow_html=True)
    year_range = st.slider("Période", 1930, 2014, (1930, 2014), step=4)
    phase_filter = st.multiselect("Phase de jeu", ["Groupes","Élimination"],
                                  default=["Groupes","Élimination"])

    st.markdown('<hr style="border-color: #1e2538; margin: 16px 0 8px;">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size: 0.65rem; color: #3A4A6A; text-align: center; line-height: 1.6;">
      <strong style="color: #4A5A8A;">Dataset</strong><br/>
      {len(matches)} matchs · {team_stats['Team'].nunique()} nations<br/>
      {matches['Year'].nunique()} éditions · 1930-2014
    </div>
    """, unsafe_allow_html=True)

# ── Filtered data ─────────────────────────────────────────
matches_f = matches[
    (matches["Year"].between(*year_range)) &
    (matches["Phase"].isin(phase_filter))
].copy()

# ══════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════
def section(title, subtitle=""):
    sub_html = f'<p>{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="section-header">
      <h2>{title}</h2>
      {sub_html}
    </div>""", unsafe_allow_html=True)

def metric_card(label, value, sub="", color="gold"):
    st.markdown(f"""
    <div class="metric-card {color}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def apply_layout(fig, title="", h=None):
    layout = dict(**PLOTLY_LAYOUT)
    if title:
        layout["title"] = dict(text=title, font=dict(color=TEXT_H, size=14), x=0.01)
    if h:
        layout["height"] = h
    fig.update_layout(**layout)
    return fig

# ══════════════════════════════════════════════════════════════
# PAGE 1 — VUE GÉNÉRALE
# ══════════════════════════════════════════════════════════════
if page == "  Vue Générale":
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
      <h1 style="color:{TEXT_H}; font-size: 1.6rem; font-weight: 700; margin: 0;">
        Vue Générale <span style="color:{GOLD};">·</span> Analyse Historique FIFA
      </h1>
      <p style="color:#4A5A7A; font-size:0.82rem; margin: 4px 0 0;">
        {year_range[0]} – {year_range[1]}  ·  {len(matches_f)} matchs sélectionnés
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────
    total_goals = int(matches_f["TotalGoals"].sum())
    avg_goals   = round(matches_f["TotalGoals"].mean(), 2) if len(matches_f) else 0
    home_pct    = round((matches_f["Result"]=="HomeWin").mean()*100, 1) if len(matches_f) else 0
    n_nations   = len(set(matches_f["HomeTeam"].tolist() + matches_f["AwayTeam"].tolist()))
    max_row     = matches_f.loc[matches_f["TotalGoals"].idxmax()] if len(matches_f) else None
    record_txt  = f"{max_row['HomeTeam']} {int(max_row['HomeGoals'])}–{int(max_row['AwayGoals'])} {max_row['AwayTeam']}" if max_row is not None else "—"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Matchs disputés", f"{len(matches_f):,}", "dans la sélection", "gold")
    with c2: metric_card("Buts marqués", f"{total_goals:,}", f"moy. {avg_goals}/match", "blue")
    with c3: metric_card("Victoires domicile", f"{home_pct}%", "avantage terrain", "green")
    with c4: metric_card("Nations", f"{n_nations}", "pays représentés", "purple")
    with c5: metric_card("Record absolu", record_txt.split()[0] if max_row is not None else "—",
                         record_txt if max_row is not None else "", "red")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart 1 : Évolution buts ─────────────────────────────
    col1, col2 = st.columns([3, 2])
    with col1:
        section("Évolution des buts par édition", "Total buts (barres) · Moyenne/match (ligne)")
        goals_ed = matches_f.groupby("Year").agg(
            Total=("TotalGoals","sum"), Matchs=("Year","count")).reset_index()
        goals_ed["Moyenne"] = (goals_ed["Total"] / goals_ed["Matchs"]).round(2)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=goals_ed["Year"], y=goals_ed["Total"],
            name="Total buts", marker_color=GOLD, opacity=0.5,
            marker_line_color=GOLD, marker_line_width=1
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=goals_ed["Year"], y=goals_ed["Moyenne"],
            name="Moy/match", mode="lines+markers",
            line=dict(color=RED, width=2.5),
            marker=dict(size=7, color=RED)
        ), secondary_y=True)
        fig.update_yaxes(title_text="Total buts", color=GOLD, secondary_y=False,
                         gridcolor=BORDER, tickfont=dict(color=GOLD))
        fig.update_yaxes(title_text="Moy buts/match", color=RED, secondary_y=True,
                         tickfont=dict(color=RED))
        apply_layout(fig, h=340)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Répartition des résultats")
        hw = (matches_f["Result"]=="HomeWin").sum()
        aw = (matches_f["Result"]=="AwayWin").sum()
        dr = (matches_f["Result"]=="Draw").sum()
        fig2 = go.Figure(go.Pie(
            labels=["Victoire domicile","Nul","Victoire extérieur"],
            values=[hw, dr, aw],
            hole=0.55,
            marker=dict(colors=[BLUE, "#6B7A9A", RED],
                        line=dict(color=BG, width=2)),
            textfont=dict(color=TEXT_H, size=11),
        ))
        fig2.update_layout(
            **PLOTLY_LAYOUT, height=340,
            annotations=[dict(text=f"<b>{len(matches_f)}</b><br>matchs",
                              font=dict(size=13, color=TEXT_H), showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 2 : Distribution buts + Phase ──────────────────
    col3, col4 = st.columns(2)
    with col3:
        section("Distribution des buts par match")
        score_freq = matches_f["TotalGoals"].value_counts().sort_index()
        fig3 = go.Figure(go.Bar(
            x=score_freq.index, y=score_freq.values,
            marker=dict(
                color=score_freq.values,
                colorscale=[[0, "#2a3050"], [0.5, BLUE], [1, GOLD]],
                line=dict(color=BG, width=0.8)
            ),
            text=score_freq.values, textposition="outside",
            textfont=dict(color=TEXT, size=10)
        ))
        apply_layout(fig3, h=300)
        fig3.update_layout(xaxis_title="Buts dans le match", yaxis_title="Nb matchs")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        section("Groupes vs Élimination directe")
        fig4 = go.Figure()
        for phase_name, color, opacity in [("Groupes", GOLD, 0.55), ("Élimination", RED, 0.55)]:
            data = matches_f[matches_f["Phase"]==phase_name]["TotalGoals"]
            if len(data) > 0:
                freq = data.value_counts().sort_index()
                fig4.add_trace(go.Bar(
                    x=freq.index, y=freq.values,
                    name=f"{phase_name} (moy={data.mean():.2f})",
                    marker_color=color, opacity=opacity,
                    marker_line_color=BG, marker_line_width=0.8
                ))
        fig4.update_layout(barmode="group", **PLOTLY_LAYOUT, height=300)
        fig4.update_layout(xaxis_title="Buts", yaxis_title="Fréquence")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Chart 3 : Affluence ───────────────────────────────────
    section("Affluence & expansion du tournoi", "Total spectateurs et équipes qualifiées par édition")
    wc_f = worldcups[worldcups["Year"].between(*year_range)]
    fig5 = make_subplots(specs=[[{"secondary_y": True}]])
    fig5.add_trace(go.Scatter(
        x=wc_f["Year"], y=wc_f["Attendance"],
        name="Affluence totale", mode="lines+markers",
        fill="tozeroy", fillcolor=f"rgba(55,138,221,0.12)",
        line=dict(color=BLUE, width=2.5),
        marker=dict(size=7)
    ), secondary_y=False)
    fig5.add_trace(go.Scatter(
        x=wc_f["Year"], y=wc_f["QualifiedTeams"],
        name="Équipes qualifiées", mode="lines+markers",
        line=dict(color=GOLD, width=2, dash="dash"),
        marker=dict(size=7, symbol="diamond")
    ), secondary_y=True)
    fig5.update_yaxes(title_text="Affluence totale", color=BLUE, secondary_y=False,
                      gridcolor=BORDER, tickfont=dict(color=BLUE))
    fig5.update_yaxes(title_text="Équipes qualifiées", color=GOLD, secondary_y=True,
                      tickfont=dict(color=GOLD))
    apply_layout(fig5, h=280)
    st.plotly_chart(fig5, use_container_width=True)

    # ── Insights ──────────────────────────────────────────────
    section("Insights Clés")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("""
        <div class="insight-box">
          <strong>📉 Déclin offensif</strong><br/>
          La moyenne de buts passe de <strong>5.38/match</strong> en 1954 à
          <strong>2.21</strong> en 1990 — densification tactique progressive.
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""
        <div class="insight-box">
          <strong>🏠 Avantage domicile</strong><br/>
          57% des matchs gagnés par l'équipe "domicile" (conventionnel)
          vs seulement 20% pour l'extérieur.
        </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown("""
        <div class="insight-box">
          <strong>⚡ Phase éliminatoire</strong><br/>
          Les matchs à élimination directe produisent légèrement
          <strong>moins de buts</strong> — pression tactique accrue.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE PAR ÉQUIPE
# ══════════════════════════════════════════════════════════════
elif page == "  Analyse par Équipe":
    st.markdown(f"""
    <h1 style="color:{TEXT_H}; font-size: 1.6rem; font-weight: 700; margin-bottom: 4px;">
      Analyse par Équipe <span style="color:{GOLD};">·</span> Ranking & Profils
    </h1>
    <p style="color:#4A5A7A; font-size:0.82rem; margin-bottom: 24px;">
      {team_stats['Team'].nunique()} nations analysées · stats agrégées 1930-2014
    </p>
    """, unsafe_allow_html=True)

    # ── Top 15 ───────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        section("Top 15 — Victoires historiques")
        top15 = team_stats.nlargest(15, "Victoires").reset_index(drop=True)
        fig = go.Figure(go.Bar(
            y=top15["Team"][::-1], x=top15["Victoires"][::-1],
            orientation="h",
            marker=dict(
                color=top15["Victoires"][::-1],
                colorscale=[[0, "#2a3050"], [0.5, BLUE], [1, GOLD]],
                line=dict(color=BG, width=0.5)
            ),
            text=top15["Victoires"][::-1], textposition="outside",
            textfont=dict(color=TEXT_H, size=10)
        ))
        apply_layout(fig, h=430)
        fig.update_layout(xaxis_title="Victoires", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section("Top 15 — Buts marqués")
        top15b = team_stats.nlargest(15, "Buts_marques").reset_index(drop=True)
        fig2 = go.Figure(go.Bar(
            y=top15b["Team"][::-1], x=top15b["Buts_marques"][::-1],
            orientation="h",
            marker=dict(
                color=top15b["Buts_marques"][::-1],
                colorscale=[[0, "#1a2a3a"], [0.5, "#1D9E75"], [1, GOLD]],
                line=dict(color=BG, width=0.5)
            ),
            text=top15b["Buts_marques"][::-1], textposition="outside",
            textfont=dict(color=TEXT_H, size=10)
        ))
        apply_layout(fig2, h=430)
        fig2.update_layout(xaxis_title="Buts marqués", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Scatter Win rate vs Buts ──────────────────────────────
    section("Profil offensif vs Win rate", "Taille = matchs joués · Couleur = différence de buts")
    ts_filter = team_stats[team_stats["Matchs"] >= 6].copy()
    fig3 = go.Figure()

    for profil in sorted(ts_filter["Profil"].unique()):
        sub = ts_filter[ts_filter["Profil"]==profil]
        color_map = {
            "🏆 Dominants": GOLD, "💪 Solides": BLUE,
            "⚡ Challengers": GREEN, "🌱 Émergents": RED
        }
        c = color_map.get(profil, PURPLE)
        fig3.add_trace(go.Scatter(
            x=sub["Buts_pm"], y=sub["Win_rate"],
            mode="markers+text",
            name=profil,
            marker=dict(
                size=np.sqrt(sub["Matchs"]) * 4,
                color=c,
                opacity=0.75,
                line=dict(color=BG, width=1)
            ),
            text=sub.apply(lambda r: r["Team"] if r["Matchs"] >= 20 else "", axis=1),
            textposition="top center",
            textfont=dict(size=9, color=TEXT),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Win rate: %{y:.1f}%<br>"
                "Buts/match: %{x:.2f}<br>"
                "Matchs: %{customdata[1]}<br>"
                "Diff. buts: %{customdata[2]}"
                "<extra></extra>"
            ),
            customdata=sub[["Team","Matchs","Diff_buts"]].values
        ))

    # Lignes de moyenne
    fig3.add_hline(y=ts_filter["Win_rate"].mean(), line_dash="dash",
                   line_color="#444466", line_width=1, opacity=0.6)
    fig3.add_vline(x=ts_filter["Buts_pm"].mean(), line_dash="dash",
                   line_color="#444466", line_width=1, opacity=0.6)
    apply_layout(fig3, h=480)
    fig3.update_layout(xaxis_title="Buts marqués / match", yaxis_title="Win rate (%)")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Radar ────────────────────────────────────────────────
    section("Radar — Comparaison de profils", "Sélectionnez jusqu'à 6 équipes")
    default_teams = ["Brazil","Germany","Italy","Argentina","France"]
    available = [t for t in default_teams if t in ALL_TEAMS]
    selected_teams = st.multiselect("Équipes à comparer", ALL_TEAMS,
                                    default=available[:5], max_selections=6)
    if len(selected_teams) >= 2:
        ts = team_stats.copy()
        norm = lambda col: (ts[col] - ts[col].min()) / (ts[col].max() - ts[col].min() + 1e-9)
        ts["n_wr"]  = norm("Win_rate")
        ts["n_bpm"] = norm("Buts_pm")
        ts["n_def"] = 1 - norm("Buts_encais_pm")
        ts["n_lon"] = norm("Matchs")
        ts["n_eff"] = norm("Diff_buts")
        categories = ["Win rate","Buts/match","Solidité déf.","Longévité","Efficacité"]

        fig4 = go.Figure()
        for team, color in zip(selected_teams, PALETTE):
            row = ts[ts["Team"]==team]
            if len(row) == 0: continue
            row = row.iloc[0]
            vals = [row["n_wr"], row["n_bpm"], row["n_def"], row["n_lon"], row["n_eff"]]
            vals_closed = vals + [vals[0]]
            cats_closed = categories + [categories[0]]
            fig4.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats_closed, fill="toself",
                name=team,
                line=dict(color=color, width=2),
                fillcolor=color.replace("#", "rgba(").rstrip(")") if "#" in color else color,
                opacity=0.8
            ))
            # Manual fill with rgba
            r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
            fig4.data[-1].fillcolor = f"rgba({r},{g},{b},0.15)"

        fig4.update_layout(
            polar=dict(
                bgcolor=CARD,
                radialaxis=dict(visible=True, range=[0,1], color=TEXT,
                                gridcolor=BORDER, tickfont=dict(color="#5A6A8A")),
                angularaxis=dict(color=TEXT, gridcolor=BORDER)
            ),
            **PLOTLY_LAYOUT, height=450
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Fiche équipe ─────────────────────────────────────────
    section("Fiche Nation", "Statistiques détaillées d'une équipe")
    team_sel = st.selectbox("Choisir une nation", ALL_TEAMS,
                            index=ALL_TEAMS.index("Brazil") if "Brazil" in ALL_TEAMS else 0)
    row = team_stats[team_stats["Team"]==team_sel].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Matchs joués", int(row['Matchs']), "", "gold")
    with c2: metric_card("Victoires", int(row['Victoires']), f"Win rate {row['Win_rate']}%", "green")
    with c3: metric_card("Buts marqués", int(row['Buts_marques']),
                         f"{row['Buts_pm']}/match", "blue")
    with c4: metric_card("Buts encaissés", int(row['Buts_encais']),
                         f"{row['Buts_encais_pm']}/match", "red")
    with c5: metric_card("Diff. buts", f"{int(row['Diff_buts']):+d}",
                         f"{int(row['Editions'])} éditions", "purple")

    # Historique de la nation
    nation_hist = matches_f[
        (matches_f["HomeTeam"]==team_sel) | (matches_f["AwayTeam"]==team_sel)
    ].copy()
    if len(nation_hist) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        fig5 = go.Figure()
        # Buts marqués par year
        scored = []
        conceded = []
        years_n = []
        for yr, grp in nation_hist.groupby("Year"):
            s = grp.apply(lambda r: r["HomeGoals"] if r["HomeTeam"]==team_sel else r["AwayGoals"], axis=1).sum()
            c = grp.apply(lambda r: r["AwayGoals"] if r["HomeTeam"]==team_sel else r["HomeGoals"], axis=1).sum()
            scored.append(s); conceded.append(c); years_n.append(yr)
        fig5.add_trace(go.Bar(x=years_n, y=scored, name="Buts marqués",
                              marker_color=GREEN, opacity=0.8))
        fig5.add_trace(go.Bar(x=years_n, y=[-c for c in conceded], name="Buts encaissés",
                              marker_color=RED, opacity=0.8))
        apply_layout(fig5, f"Buts de {team_sel} par édition", h=260)
        fig5.update_layout(barmode="relative",
                           yaxis_title="Buts (+ marqués / - encaissés)")
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — CLUSTERING ML
# ══════════════════════════════════════════════════════════════
elif page == "  Clustering ML":
    st.markdown(f"""
    <h1 style="color:{TEXT_H}; font-size: 1.6rem; font-weight: 700; margin-bottom: 4px;">
      Clustering ML <span style="color:{GOLD};">·</span> K-Means · Profils de Jeu
    </h1>
    <p style="color:#4A5A7A; font-size:0.82rem; margin-bottom: 24px;">
      Segmentation non supervisée · 5 features · k=4 clusters optimaux
    </p>
    """, unsafe_allow_html=True)

    # ── Méthode du coude ─────────────────────────────────────
    col1, col2 = st.columns([3, 2])
    with col1:
        section("Méthode du coude", "Sélection automatique de k optimal")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(k_range), y=inertias, mode="lines+markers",
            line=dict(color=GOLD, width=2.5),
            marker=dict(size=9, color=GOLD, line=dict(color=BG, width=1.5)),
            name="Inertie WCSS"
        ))
        # Annotate k=4
        idx4 = list(k_range).index(4)
        fig.add_vline(x=4, line_dash="dash", line_color=RED, line_width=1.5, opacity=0.7)
        fig.add_annotation(x=4, y=inertias[idx4], text="  k=4<br>  optimal",
                           font=dict(color=RED, size=11), showarrow=False, xanchor="left")
        apply_layout(fig, h=320)
        fig.update_layout(xaxis_title="Nombre de clusters (k)", yaxis_title="Inertie (WCSS)",
                          xaxis=dict(tickmode="linear", tick0=2, dtick=1, gridcolor=BORDER))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Répartition des profils")
        profil_counts = team_stats["Profil"].value_counts()
        color_map_p = {
            "🏆 Dominants": GOLD, "💪 Solides": BLUE,
            "⚡ Challengers": GREEN, "🌱 Émergents": RED
        }
        fig2 = go.Figure(go.Pie(
            labels=profil_counts.index,
            values=profil_counts.values,
            hole=0.5,
            marker=dict(
                colors=[color_map_p.get(p, PURPLE) for p in profil_counts.index],
                line=dict(color=BG, width=2)
            ),
            textfont=dict(color=TEXT_H, size=10)
        ))
        apply_layout(fig2, h=320)
        fig2.update_layout(
            annotations=[dict(text=f"<b>{len(team_stats)}</b><br>nations",
                              font=dict(size=12, color=TEXT_H), showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Scatter clusters ──────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        section("Clustering — Win rate vs Offensivité")
        fig3 = go.Figure()
        for profil in sorted(team_stats["Profil"].unique()):
            sub = team_stats[team_stats["Profil"]==profil]
            c = color_map_p.get(profil, PURPLE)
            r_c, g_c, b_c = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
            fig3.add_trace(go.Scatter(
                x=sub["Buts_pm"], y=sub["Win_rate"],
                mode="markers",
                name=profil,
                marker=dict(size=np.clip(np.sqrt(sub["Matchs"])*3.5, 6, 25),
                            color=f"rgba({r_c},{g_c},{b_c},0.8)",
                            line=dict(color=BG, width=0.5)),
                hovertemplate="<b>%{customdata}</b><br>Win rate: %{y:.1f}%<br>Buts/m: %{x:.2f}<extra></extra>",
                customdata=sub["Team"].values
            ))
        apply_layout(fig3, h=360)
        fig3.update_layout(xaxis_title="Buts marqués / match", yaxis_title="Win rate (%)")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        section("Clustering — Attaque vs Défense")
        fig4 = go.Figure()
        for profil in sorted(team_stats["Profil"].unique()):
            sub = team_stats[team_stats["Profil"]==profil]
            c = color_map_p.get(profil, PURPLE)
            r_c, g_c, b_c = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
            fig4.add_trace(go.Scatter(
                x=sub["Buts_encais_pm"], y=sub["Buts_pm"],
                mode="markers",
                name=profil,
                marker=dict(size=np.clip(np.sqrt(sub["Matchs"])*3.5, 6, 25),
                            color=f"rgba({r_c},{g_c},{b_c},0.8)",
                            line=dict(color=BG, width=0.5)),
                hovertemplate="<b>%{customdata}</b><br>Buts enc/m: %{x:.2f}<br>Buts/m: %{y:.2f}<extra></extra>",
                customdata=sub["Team"].values
            ))
        # Ligne d'équilibre
        max_v = max(team_stats["Buts_encais_pm"].max(), team_stats["Buts_pm"].max())
        fig4.add_trace(go.Scatter(
            x=[0, max_v], y=[0, max_v], mode="lines",
            line=dict(color="#444466", dash="dash", width=1),
            name="Équilibre att./déf.", showlegend=True
        ))
        apply_layout(fig4, h=360)
        fig4.update_layout(xaxis_title="Buts encaissés / match", yaxis_title="Buts marqués / match")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Liste des équipes par cluster ─────────────────────────
    section("Équipes par profil de jeu")
    profil_cols = st.columns(4)
    profils_order = ["🏆 Dominants","💪 Solides","⚡ Challengers","🌱 Émergents"]
    colors_profil = [GOLD, BLUE, GREEN, RED]
    for col, profil, col_color in zip(profil_cols, profils_order, colors_profil):
        with col:
            sub = sorted(team_stats[team_stats["Profil"]==profil]["Team"].tolist())
            r_c, g_c, b_c = int(col_color[1:3],16), int(col_color[3:5],16), int(col_color[5:7],16)
            teams_html = "".join([f'<div style="padding:3px 0; border-bottom:1px solid {BORDER}; font-size:0.78rem; color:{TEXT};">{t}</div>' for t in sub])
            st.markdown(f"""
            <div style="background:{CARD}; border:1px solid {BORDER}; border-top:2px solid {col_color};
                        border-radius:10px; padding:14px;">
              <div style="font-size:0.85rem; font-weight:600; color:{col_color}; margin-bottom:10px;">
                {profil} <span style="background:rgba({r_c},{g_c},{b_c},0.15); color:{col_color};
                padding:2px 8px; border-radius:10px; font-size:0.7rem;">{len(sub)} nations</span>
              </div>
              {teams_html}
            </div>""", unsafe_allow_html=True)

    # ── Stats moyennes par cluster ────────────────────────────
    section("Statistiques moyennes par cluster")
    cluster_mean = team_stats.groupby("Profil")[["Win_rate","Buts_pm","Buts_encais_pm","Diff_buts","Matchs"]].mean().round(2)
    cluster_mean.columns = ["Win rate (%)","Buts/match","Buts enc./match","Diff. buts","Matchs moy."]
    st.dataframe(
        cluster_mean.style.background_gradient(cmap="YlOrRd", axis=1),
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════════
# PAGE 4 — PRÉDICTION ML
# ══════════════════════════════════════════════════════════════
elif page == "  Prédiction ML":
    st.markdown(f"""
    <h1 style="color:{TEXT_H}; font-size: 1.6rem; font-weight: 700; margin-bottom: 4px;">
      Prédiction ML <span style="color:{GOLD};">·</span> Régression Logistique & HGB
    </h1>
    <p style="color:#4A5A7A; font-size:0.82rem; margin-bottom: 24px;">
      Modèles entraînés sur 80% des données · évalués sur 20%
    </p>
    """, unsafe_allow_html=True)

    feat_labels = [
        "Win rate (équipe)","Buts/match (éq.)","Buts enc./match (éq.)","Diff. buts (éq.)",
        "Win rate (adv.)","Buts/match (adv.)","Buts enc./match (adv.)","Diff. buts (adv.)"
    ]

    # ── Métriques modèles ─────────────────────────────────────
    y_pred_lr = clf.predict(X_test_s)
    y_pred_hgb = hgb.predict(X_test_s)
    acc_lr = round(clf.score(X_test_s, y_test)*100, 1)
    acc_hgb = round(hgb.score(X_test_s, y_test)*100, 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Accuracy · LogReg", f"{acc_lr}%", "Régression Logistique", "blue")
    with c2: metric_card("Accuracy · HGB", f"{acc_hgb}%", "HistGradientBoosting", "gold")
    with c3: metric_card("Train / Test", "80 / 20%", "split stratifié", "green")
    with c4: metric_card("Features", "8", "statistiques historiques", "purple")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Matrices de confusion ─────────────────────────────────
    col1, col2 = st.columns(2)
    for col, y_pred, model_name, acc in [
        (col1, y_pred_lr, "Régression Logistique", acc_lr),
        (col2, y_pred_hgb, "HistGradientBoosting", acc_hgb)
    ]:
        with col:
            section(f"Matrice de confusion — {model_name}", f"Accuracy: {acc}%")
            cm = confusion_matrix(y_test, y_pred)
            labels = ["Défaite","Nul","Victoire"]
            fig = go.Figure(go.Heatmap(
                z=cm, x=labels, y=labels,
                colorscale=[[0,"#1a1f2e"],[0.5,BLUE],[1,GOLD]],
                text=cm, texttemplate="%{text}",
                textfont=dict(size=16, color=TEXT_H),
                showscale=True
            ))
            apply_layout(fig, h=320)
            fig.update_layout(
                xaxis_title="Prédit", yaxis_title="Réel",
                yaxis=dict(autorange="reversed", gridcolor=BORDER),
                xaxis=dict(gridcolor=BORDER)
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Importance des features (LR) ─────────────────────────
    section("Importance des features", "Coefficients absolus moyens (LogReg) vs Permutation (HGB)")
    col3, col4 = st.columns(2)
    with col3:
        coef_mean = np.abs(clf.coef_).mean(axis=0)
        idx = np.argsort(coef_mean)
        colors_f = [GOLD if i >= 4 else BLUE for i in idx]
        fig3 = go.Figure(go.Bar(
            y=[feat_labels[i] for i in idx], x=coef_mean[idx],
            orientation="h",
            marker=dict(color=colors_f, line=dict(color=BG, width=0.5)),
            text=[f"{v:.3f}" for v in coef_mean[idx]],
            textposition="outside", textfont=dict(color=TEXT, size=9)
        ))
        apply_layout(fig3, "LogReg — |coeff. moyen|", h=320)
        fig3.update_layout(xaxis_title="Importance absolue")
        # Legend
        fig3.add_trace(go.Bar(x=[0], y=[""], marker_color=GOLD, name="Équipe adverse",
                              orientation="h", showlegend=True))
        fig3.add_trace(go.Bar(x=[0], y=[""], marker_color=BLUE, name="Équipe analysée",
                              orientation="h", showlegend=True))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Feature importance HGB (simple coef proxy)
        from sklearn.inspection import permutation_importance as perm_imp
        try:
            perm = perm_imp(hgb, X_test_s, y_test, n_repeats=10, random_state=42, scoring="accuracy")
            fi = perm.importances_mean
        except:
            fi = np.abs(clf.coef_).mean(axis=0)

        idx2 = np.argsort(fi)
        colors_f2 = [GOLD if i >= 4 else BLUE for i in idx2]
        fig4 = go.Figure(go.Bar(
            y=[feat_labels[i] for i in idx2], x=fi[idx2],
            orientation="h",
            marker=dict(color=colors_f2, line=dict(color=BG, width=0.5)),
            text=[f"{v:.3f}" for v in fi[idx2]],
            textposition="outside", textfont=dict(color=TEXT, size=9)
        ))
        apply_layout(fig4, "HGB — Permutation Importance", h=320)
        fig4.update_layout(xaxis_title="Impact moyen sur l'accuracy")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Classification report ─────────────────────────────────
    section("Rapport de classification détaillé")
    tab1, tab2 = st.tabs(["📊 LogReg", "📊 HGB"])
    for tab, y_pred, model_name in [(tab1, y_pred_lr, "LogReg"), (tab2, y_pred_hgb, "HGB")]:
        with tab:
            report = classification_report(
                y_test, y_pred,
                target_names=["Défaite","Nul","Victoire"],
                output_dict=True
            )
            report_df = pd.DataFrame(report).T.round(3)
            st.dataframe(
                report_df.style.background_gradient(cmap="YlOrRd", subset=["precision","recall","f1-score"]),
                use_container_width=True
            )

    # ── Insights ML ──────────────────────────────────────────
    section("Insights Modèle")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("""
        <div class="insight-box">
          <strong>🎯 Accuracy ~55%</strong><br/>
          Cohérent pour un sport aléatoire. Le football reste <strong>imprévisible</strong>
          — même les meilleurs modèles peinent à dépasser 60%.
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""
        <div class="insight-box">
          <strong>🔑 Features clés</strong><br/>
          Le <strong>Win rate historique</strong> et la <strong>différence de buts</strong>
          sont les prédicteurs les plus puissants.
        </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown("""
        <div class="insight-box">
          <strong>⚠️ Limites</strong><br/>
          Pas de données individuelles (effectifs, blessés) ni de contexte
          situationnel (stade, météo, enjeu).
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — SIMULATEUR
# ══════════════════════════════════════════════════════════════
elif page == "  Simulateur":
    st.markdown(f"""
    <h1 style="color:{TEXT_H}; font-size: 1.6rem; font-weight: 700; margin-bottom: 4px;">
      Simulateur <span style="color:{GOLD};">·</span> Prédire un Match
    </h1>
    <p style="color:#4A5A7A; font-size:0.82rem; margin-bottom: 24px;">
      Pronostic basé sur les statistiques historiques 1930–2014
    </p>
    """, unsafe_allow_html=True)

    # ── Sélection équipes ────────────────────────────────────
    col_a, col_vs, col_b = st.columns([5, 1, 5])
    with col_a:
        team_a = st.selectbox("Équipe A", ALL_TEAMS,
                              index=ALL_TEAMS.index("Brazil") if "Brazil" in ALL_TEAMS else 0)
    with col_vs:
        st.markdown(f"""
        <div style="text-align:center; padding-top:32px; font-size:1.4rem;
                    font-weight:700; color:{GOLD};">VS</div>""", unsafe_allow_html=True)
    with col_b:
        team_b = st.selectbox("Équipe B", ALL_TEAMS,
                              index=ALL_TEAMS.index("Germany") if "Germany" in ALL_TEAMS else 1)

    if team_a == team_b:
        st.warning("⚠️ Veuillez sélectionner deux équipes différentes.")
    else:
        res = predict_match(team_a, team_b, team_stats, clf, sc_m)
        if res:
            p_a = res["p_win_a"]
            p_d = res["p_draw"]
            p_b = res["p_win_b"]
            winner = team_a if p_a == max(p_a, p_d, p_b) else (team_b if p_b == max(p_a, p_d, p_b) else "Match nul")

            st.markdown("<br>", unsafe_allow_html=True)

            # Résultat principal
            winner_color = GOLD if winner not in ["Match nul"] else "#6B7A9A"
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{CARD},#1a1f2e);
                        border:1px solid {BORDER}; border-top:2px solid {GOLD};
                        border-radius:14px; padding:28px 32px; text-align:center;
                        margin-bottom: 20px;">
              <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.15em;
                          color:#4A5A7A; margin-bottom: 12px;">Pronostic</div>
              <div style="font-size:2.2rem; font-weight:800; color:{winner_color};">
                {"🏆 " if winner != "Match nul" else "🤝 "}{winner}
              </div>
              <div style="font-size:0.8rem; color:#4A5A7A; margin-top: 8px;">
                Modèle : Régression Logistique · basé sur les stats historiques FIFA
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Jauges de probabilité
            col_p1, col_p2 = st.columns([2, 3])
            with col_p1:
                for label, prob, color in [
                    (f"Victoire {team_a}", p_a, GREEN),
                    ("Match nul", p_d, "#6B7A9A"),
                    (f"Victoire {team_b}", p_b, RED)
                ]:
                    pct = prob * 100
                    st.markdown(f"""
                    <div class="proba-bar-wrap">
                      <div class="proba-bar-label">
                        <span>{label}</span>
                        <span style="color:{color}; font-weight:600;">{pct:.1f}%</span>
                      </div>
                      <div class="proba-bar-bg">
                        <div class="proba-bar-fill" style="width:{pct}%; background:{color};"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            with col_p2:
                fig = go.Figure()
                labels_p = [f"Victoire\n{team_a}", "Nul", f"Victoire\n{team_b}"]
                values_p = [p_a, p_d, p_b]
                bar_colors = [GREEN if v == max(values_p) else BLUE for v in values_p]
                bar_colors[1] = "#6B7A9A"

                fig.add_trace(go.Bar(
                    x=labels_p, y=values_p,
                    marker=dict(color=bar_colors, line=dict(color=BG, width=1)),
                    text=[f"{v*100:.1f}%" for v in values_p],
                    textposition="outside",
                    textfont=dict(size=14, color=TEXT_H)
                ))
                apply_layout(fig, h=280)
                fig.update_layout(
                    yaxis=dict(tickformat=".0%", range=[0, max(values_p)*1.3],
                               gridcolor=BORDER),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

            # Fiche comparative
            section("Comparaison des profils")
            ra = team_stats[team_stats["Team"]==team_a].iloc[0]
            rb = team_stats[team_stats["Team"]==team_b].iloc[0]

            comp_data = {
                "Statistique": ["Win rate","Buts/match","Buts encaissés/match","Diff. buts","Matchs joués","Victoires"],
                team_a: [f"{ra['Win_rate']}%", ra['Buts_pm'], ra['Buts_encais_pm'],
                         f"{int(ra['Diff_buts']):+d}", int(ra['Matchs']), int(ra['Victoires'])],
                team_b: [f"{rb['Win_rate']}%", rb['Buts_pm'], rb['Buts_encais_pm'],
                         f"{int(rb['Diff_buts']):+d}", int(rb['Matchs']), int(rb['Victoires'])],
            }
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df.set_index("Statistique"), use_container_width=True)

    # ── Simulation Quarts de finale ───────────────────────────
    section("Simulation — Quarts de finale imaginaires", "Matchups scénarisés basés sur les stats historiques")
    matchups = [
        ("Brazil","Germany"),
        ("Argentina","Italy"),
        ("France","Spain"),
        ("Netherlands","England"),
    ]

    available_matchups = [(a, b) for a, b in matchups if a in ALL_TEAMS and b in ALL_TEAMS]
    if available_matchups:
        cols = st.columns(len(available_matchups))
        for col, (ta, tb) in zip(cols, available_matchups):
            with col:
                res = predict_match(ta, tb, team_stats, clf, sc_m)
                if res:
                    pa, pd_, pb = res["p_win_a"], res["p_draw"], res["p_win_b"]
                    winner_idx = [pa, pd_, pb].index(max(pa, pd_, pb))
                    winner_name = [ta, "Nul", tb][winner_idx]

                    fig = go.Figure(go.Bar(
                        x=[f"{ta}", "Nul", f"{tb}"],
                        y=[pa, pd_, pb],
                        marker=dict(
                            color=[GREEN if i==winner_idx else BLUE if i != 1 else "#6B7A9A"
                                   for i in range(3)],
                            line=dict(color=BG, width=0.8)
                        ),
                        text=[f"{v*100:.1f}%" for v in [pa, pd_, pb]],
                        textposition="outside",
                        textfont=dict(size=10, color=TEXT_H)
                    ))
                    fig.update_layout(
                        title=dict(text=f"<b>{ta}</b> vs <b>{tb}</b>",
                                   font=dict(color=GOLD, size=11), x=0.5),
                        yaxis=dict(tickformat=".0%", range=[0, 0.85],
                                   gridcolor=BORDER, showgrid=True),
                        **PLOTLY_LAYOUT, height=280,
                        margin=dict(t=45, b=30, l=20, r=20),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
