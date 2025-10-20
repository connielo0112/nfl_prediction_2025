"""
NFL Game Predictor 2025 
"""

import os, datetime, warnings, requests
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

# Fetches one year’s worth of data from ESPN’s API
def fetch_historical_games(year):
    """
    Fetch completed NFL games for a given calendar year from ESPN API."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?limit=1000&dates={year}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"Fetch error {year}: {e}")
        return pd.DataFrame()

    # Parse JSON response
    events = r.json().get("events", [])
    rows = []

    # Loop through each event/game
    for ev in events:
        comp = ev.get("competitions", [{}])[0]

        # Skip games not yet completed
        if not comp.get("status", {}).get("type", {}).get("completed"):
            continue

        # Extract teams (home vs away)
        teams = comp.get("competitors", [])
        if len(teams) != 2:
            continue
        home = next((t for t in teams if t.get("homeAway") == "home"), {})
        away = next((t for t in teams if t.get("homeAway") == "away"), {})

        # Store essential info in a list of dicts
        rows.append({
            "date": ev.get("date"),
            "week": (ev.get("week", {}) or {}).get("number"),
            "home_team": home.get("team", {}).get("displayName"),
            "away_team": away.get("team", {}).get("displayName"),
            "home_score": int(home.get("score") or 0),
            "away_score": int(away.get("score") or 0),
            "home_win": int((home.get("score") or "0") > (away.get("score") or "0")),
        })
    return pd.DataFrame(rows)

def get_full_history():
    """Fetch both last season and current season for up-to-date training."""
    this_year = datetime.date.today().year
    dfs = []

    # Download data for both this year and previous year
    for yr in [this_year - 1, this_year]:
        df = fetch_historical_games(yr)
        if not df.empty:
            dfs.append(df)

    # Combine all seasons into one big DataFrame
    hist = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    # Clean + format
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["home_team", "away_team"])
    print(f"Loaded {len(hist)} total completed games across {len(dfs)} years.")
    return hist

# Featre engineering: calculate team stats
def calculate_team_stats(games_df):
    """Compute win%, avg scored/allowed, and point diff per team."""
    team_stats = {}

    # Get all unique team names that appeared as home or away
    teams = pd.concat([games_df["home_team"], games_df["away_team"]]).unique()

    for t in teams:
        # Split into home and away games for this team
        hg = games_df[games_df["home_team"] == t]
        ag = games_df[games_df["away_team"] == t]
        total = len(hg) + len(ag)
        if total == 0:
            continue

        # Wins: home_wins from home team perspective + away_wins when team wins away
        home_wins = hg["home_win"].sum()
        away_wins = (ag["away_score"] > ag["home_score"]).sum()
        total_wins = home_wins + away_wins

        # Points scored and allowed
        pts_for = hg["home_score"].sum() + ag["away_score"].sum()
        pts_against = hg["away_score"].sum() + ag["home_score"].sum()
        
        # Save averages
        team_stats[t] = {
            "win_pct": total_wins / total,
            "avg_scored": pts_for / total,
            "avg_allowed": pts_against / total,
            "point_diff": (pts_for - pts_against) / total
        }
    print(f"Stats computed for {len(team_stats)} teams.")
    return team_stats

# Train Model
def train_model(games_df, team_stats):
    """Build training dataset and train Gradient Boosting model."""
    X, y = [], []

    # Build feature vectors for each completed game
    for _, g in games_df.iterrows():
        h, a = g["home_team"], g["away_team"]

        # skip if stats missing
        if h not in team_stats or a not in team_stats:
            continue
        hs, as_ = team_stats[h], team_stats[a]

        # Create 8 numeric features + 1 home-field bias
        X.append([
            hs["win_pct"], as_["win_pct"],
            hs["avg_scored"], as_["avg_scored"],
            hs["avg_allowed"], as_["avg_allowed"],
            hs["point_diff"], as_["point_diff"],
            1.0  # home field
        ])
        y.append(g["home_win"]) # label: 1 if home wins
    
    # Convert to NumPy arrays for sklearn
    X, y = np.array(X), np.array(y)

    # Split randomly into 80% train, 20% test for quick evaluation
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Gradient Boosting model
    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42)

    # Fit the model
    model.fit(Xtr, ytr)

    # Measure performance on hold-out data
    acc = accuracy_score(yte, model.predict(Xte))
    print(f"Model trained. Accuracy on hold-out: {acc:.1%}\n")
    return model

# Predict upcoming games
def fetch_upcoming_games():
    """Return games not yet completed from ESPN current scoreboard."""
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"Could not fetch upcoming: {e}")
        return pd.DataFrame()
    upcoming = []

    # Parse ESPN JSON
    for ev in r.json().get("events", []):
        comp = ev.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {})

        # Skip completed games
        if status.get("completed"):
            continue

        teams = comp.get("competitors", [])
        if len(teams) != 2:
            continue
        home = next((t for t in teams if t["homeAway"] == "home"), {})
        away = next((t for t in teams if t["homeAway"] == "away"), {})
        upcoming.append({
            "date": ev.get("date"),
            "week": (ev.get("week", {}) or {}).get("number"),
            "home_team": home.get("team", {}).get("displayName"),
            "away_team": away.get("team", {}).get("displayName")
        })
    df = pd.DataFrame(upcoming)
    print(f"Found {len(df)} upcoming games:")
    return df

def predict_games(model, upcoming_df, team_stats):
    """Generate predictions for today's and future games."""
    if upcoming_df.empty:
        print("No games to predict right now.\n")
        return pd.DataFrame()
    
    preds = []

    for _, g in upcoming_df.iterrows():
        h, a = g["home_team"], g["away_team"]
        # Only predict if both teams exist in our stats
        if h not in team_stats or a not in team_stats:
            continue
        hs, as_ = team_stats[h], team_stats[a]

        # Same feature order as training step
        features = np.array([[
            hs["win_pct"], as_["win_pct"],
            hs["avg_scored"], as_["avg_scored"],
            hs["avg_allowed"], as_["avg_allowed"],
            hs["point_diff"], as_["point_diff"],
            1.0
        ]])

        # Get model probabilities (class 0 = loss, class 1 = win)
        proba = model.predict_proba(features)[0]

        # Winner is whichever side has higher probability
        winner = h if proba[1] >= 0.5 else a
        
        preds.append({
            "date": g["date"],
            "week": g["week"],
            "home_team": h,
            "away_team": a,
            "predicted_winner": winner,
            "confidence": round(float(max(proba)), 3)
        })
        print(f"{h} vs {a} → {winner} ({max(proba):.1%})")

    # Save predictions to CSV for later tracking
    out = pd.DataFrame(preds)
    if not out.empty:
        out.to_csv("predictions_latest.csv", index=False)
    return out

# Main execution
def main():
    print("="*70)
    print("NFL GAME PREDICTOR 2025 (Auto-updated)")
    print("="*70, "\n")

    hist = get_full_history()
    if hist.empty:
        print("No historical data — check connection or off-season.")
        return

    team_stats = calculate_team_stats(hist)
    model = train_model(hist, team_stats)

    upcoming = fetch_upcoming_games()
    preds = predict_games(model, upcoming, team_stats)

    # Save training snapshot so model can be reproduced later
    os.makedirs("data_snapshots", exist_ok=True)
    hist.to_csv(f"data_snapshots/historical_{datetime.date.today()}.csv", index=False)
    print("\nDone.")

if __name__ == "__main__":
    main()