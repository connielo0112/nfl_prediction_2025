# NFL Game Predictor 2025

This project predicts the outcomes of upcoming NFL games using recent team performance statistics.  
It automatically fetches data from ESPN’s public NFL API, trains a machine learning model, and outputs predictions with confidence levels.


## Overview

**Goal:**  
Predict which team is more liely to win an upcoming NFL game based on past team performance.

**Methodology:**  
1. **Fetch historical data** — Retrieves the latest completed games from the ESPN API for the current and previous seasons.  
2. **Feature engineering** — Calculates average performance metrics for each team (win rate, offense, defense, etc.).  
3. **Model training** — Uses a Gradient Boosting Classifier to learn patterns from past results.  
4. **Prediction** — Applies the trained model to upcoming games to predict winners and confidence levels.  
5. **Output** — Saves a `predictions_latest.csv` file with the results.



## Start
### 1. Clone the repository
```bash
git clone https://github.com/yourusername/nfl-game-predictor-2025.git
cd nfl-game-predictor-2025
```

### (Optional but recommended) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the predictor
```bash
python nfl_prediction.py
```



## Model Choice: Gradient Boosting Classifier
- **Accurate for tabular data** — Works well on small to medium-sized datasets with mixed numeric features.  
- **Robust to feature scale differences** — No need for heavy preprocessing or normalization.  
- **Able to capture nonlinear patterns** — Important because NFL outcomes depend on complex, non-linear team interactions.  
- **Interpretable** — We can examine feature importance (e.g., which stats most affect predictions).



## Features Used

Each game is represented by **9 numeric features** — describing the relative strengths of both teams and home-field advantage.

| Feature | Description |
|----------|-------------|
| `home_win_pct` | Home team’s overall win percentage |
| `away_win_pct` | Away team’s overall win percentage |
| `home_avg_scored` | Average points scored by home team |
| `away_avg_scored` | Average points scored by away team |
| `home_avg_allowed` | Average points allowed by home team |
| `away_avg_allowed` | Average points allowed by away team |
| `home_point_diff` | Average margin of victory (home team) |
| `away_point_diff` | Average margin of victory (away team) |
| `home_field` | Constant = 1.0 (represents home-field advantage) |

These features capture both **offensive** and **defensive** strengths, and provide a balanced comparison between the two teams.



## Model Output Example
```bash
======================================================================
NFL GAME PREDICTOR 2025 (Auto-updated)
====================================================================== 

Loaded 520 total completed games across 2 years.
Stats computed for 34 teams.
Model trained. Accuracy on hold-out: 61.5%

Found 2 upcoming games:
Detroit Lions vs Tampa Bay Buccaneers → Detroit Lions (50.5%)
Seattle Seahawks vs Houston Texans → Houston Texans (99.1%)
```
- The **predicted winner** is the team expected to win.  
- The **confidence (%)** represents how certain the model is in that prediction.  



## Future Improvements
- Incorporate **player-level** stats (e.g., QB rating, rushing yards).  
- Include **real-time updates** (injuries, trades, weather).  
- Explore **XGBoost or LightGBM** for faster and possibly more accurate results.  
- Add **feature importance visualization**.



**Author:** Connie Lo  
**Date:** October 2025
