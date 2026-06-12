"""
data_engine.py — World Cup Intelligence Platform
Toute la logique données + ML, séparée du UI
"""
 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
 
# ── Unification des noms historiques ─────────────────────────────────────────
TEAM_UNIFY = {
    "Germany FR":        "Germany",
    "Czechoslovakia":    "Czech Republic",
    "Soviet Union":      "Russia",
    "Yugoslavia":        "Serbia",
    "Dutch East Indies": "Indonesia",
}
 
PROFIL_ORDER  = ["🏆 Dominants", "💪 Solides", "⚡ Challengers", "🌱 Émergents"]
FEATURES_CL   = ["Win_rate", "Buts_pm", "Buts_encais_pm", "Diff_buts", "Matchs"]
FEATURES_M    = ["Win_rate", "Buts_pm", "Buts_encais_pm", "Diff_buts",
                 "Opp_wr",   "Opp_bpm", "Opp_bencais",    "Opp_diff"]
 
 
# ── 1. Chargement & nettoyage ────────────────────────────────────────────────
def build_dataframes(path="world_cup_results.xlsx"):
    xl        = pd.read_excel(path, sheet_name=None)
    matches   = xl["WorldCupMatches"].copy()
    tableau   = xl["World Cup - Tableau format"].copy()
    worldcups = xl["WorldCups"].copy()
 
    # Unification des noms
    for df in [matches, tableau]:
        for col in ["HomeTeam", "AwayTeam", "Team", "Opponent"]:
            if col in df.columns:
                df[col] = df[col].replace(TEAM_UNIFY)
    worldcups["Winner"]     = worldcups["Winner"].replace(TEAM_UNIFY)
    worldcups["Runners-Up"] = worldcups["Runners-Up"].replace(TEAM_UNIFY)
 
    # Features dérivées — matches
    matches["TotalGoals"] = matches["HomeGoals"] + matches["AwayGoals"]
    matches["Margin"]     = abs(matches["HomeGoals"] - matches["AwayGoals"])
    matches["Result"]     = matches.apply(
        lambda r: "HomeWin" if r["HomeGoals"] > r["AwayGoals"]
        else ("AwayWin" if r["HomeGoals"] < r["AwayGoals"] else "Draw"), axis=1)
    matches["Phase"] = matches["Round"].apply(
        lambda r: "Groupes"
        if any(k in str(r).lower() for k in ["group", "first round", "preliminary"])
        else "Élimination")
 
    # Features dérivées — tableau
    tableau["Result"] = tableau.apply(
        lambda r: "Win"  if r["Team G"] > r["Opponent G"]
        else ("Loss" if r["Team G"] < r["Opponent G"] else "Draw"), axis=1)
 
    # Nettoyage Attendance
    def clean_att(v):
        if isinstance(v, (int, float)): return float(v)
        return float(str(v).replace(".", "").replace(",", "."))
    worldcups["Attendance"] = worldcups["Attendance"].apply(clean_att)
 
    return matches, tableau, worldcups
 
 
# ── 2. Statistiques par équipe ───────────────────────────────────────────────
def build_team_stats(tableau):
    g = tableau.groupby("Team").agg(
        Matchs        = ("Result", "count"),
        Victoires     = ("Result", lambda x: (x == "Win").sum()),
        Nuls          = ("Result", lambda x: (x == "Draw").sum()),
        Defaites      = ("Result", lambda x: (x == "Loss").sum()),
        Buts_marques  = ("Team G",     "sum"),
        Buts_encais   = ("Opponent G", "sum"),
        Editions      = ("Year",       "nunique"),
    ).reset_index()
    g["Win_rate"]       = (g["Victoires"] / g["Matchs"] * 100).round(1)
    g["Buts_pm"]        = (g["Buts_marques"] / g["Matchs"]).round(2)
    g["Buts_encais_pm"] = (g["Buts_encais"]  / g["Matchs"]).round(2)
    g["Diff_buts"]      = g["Buts_marques"] - g["Buts_encais"]
    return g
 
 
# ── 3. Clustering K-Means ────────────────────────────────────────────────────
def build_clusters(team_stats, k=4):
    X      = team_stats[FEATURES_CL].copy()
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
 
    # Méthode du coude
    k_range = range(2, 10)
    inertias = [KMeans(n_clusters=ki, random_state=42, n_init=10).fit(X_s).inertia_
                for ki in k_range]
 
    # Clustering final
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    team_stats = team_stats.copy()
    team_stats["Cluster"] = km.fit_predict(X_s)
 
    # Nommage par win_rate décroissant
    means     = team_stats.groupby("Cluster")["Win_rate"].mean().sort_values(ascending=False)
    label_map = {c: PROFIL_ORDER[i] for i, c in enumerate(means.index)}
    team_stats["Profil"] = team_stats["Cluster"].map(label_map)
 
    return team_stats, k_range, inertias
 
 
# ── 4. Modèle ML (Régression Logistique uniquement) ─────────────────────────
def build_ml_model(team_stats, tableau):
    stat_cols = ["Team", "Win_rate", "Buts_pm", "Buts_encais_pm", "Diff_buts"]
 
    df = tableau.merge(team_stats[stat_cols], on="Team")
    df = df.merge(
        team_stats[stat_cols].rename(columns={
            "Team":           "Opponent",
            "Win_rate":       "Opp_wr",
            "Buts_pm":        "Opp_bpm",
            "Buts_encais_pm": "Opp_bencais",
            "Diff_buts":      "Opp_diff",
        }),
        on="Opponent",
    )
    df["label"] = df["Result"].map({"Win": 2, "Draw": 1, "Loss": 0})
    df = df.dropna(subset=FEATURES_M + ["label"])
 
    X, y = df[FEATURES_M], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
 
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
 
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    clf.fit(X_tr_s, y_tr)
 
    return clf, sc, X_te_s, y_te, FEATURES_M
 
 
# ── 5. Prédiction match ──────────────────────────────────────────────────────
def predict_match(team_a, team_b, team_stats, clf, sc):
    ra = team_stats[team_stats["Team"] == team_a]
    rb = team_stats[team_stats["Team"] == team_b]
    if ra.empty or rb.empty:
        return None
    ra, rb = ra.iloc[0], rb.iloc[0]
 
    vec = [[ra["Win_rate"], ra["Buts_pm"], ra["Buts_encais_pm"], ra["Diff_buts"],
            rb["Win_rate"], rb["Buts_pm"], rb["Buts_encais_pm"], rb["Diff_buts"]]]
    proba = clf.predict_proba(sc.transform(vec))[0]
 
    return {"p_win_a": proba[2], "p_draw": proba[1], "p_win_b": proba[0]}
 
