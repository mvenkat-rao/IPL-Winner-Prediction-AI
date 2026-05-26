"""
model.py  — Advanced IPL prediction model trainer (XGBoost Engine)
Trains four models and saves them to models/
  1. winner_model.pkl       — pre-match winner classifier (XGBoost)
  2. win_prob_model.pkl     — Live win-probability classifier (XGBoost)
  3. score_model_first.pkl  — first-innings score regressor (GradientBoosting)
  4. score_model_second.pkl — second-innings score regressor (GradientBoosting)

Run:  python model.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, mean_absolute_error

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = 'dataset'
MODEL_DIR  = 'models'
MIN_SEASON = 2018

os.makedirs(MODEL_DIR, exist_ok=True)

TEAMS = {
    'csk': 0, 'chennai super kings': 0,
    'mi': 1, 'mumbai indians': 1,
    'rcb': 2, 'royal challengers bangalore': 2, 'royal challengers bengaluru': 2,
    'kkr': 3, 'kolkata knight riders': 3,
    'srh': 4, 'sunrisers hyderabad': 4, 'deccan chargers': 4,
    'dc': 5, 'delhi capitals': 5, 'delhi daredevils': 5,
    'pbks': 6, 'punjab kings': 6, 'kings xi punjab': 6,
    'rr': 7, 'rajasthan royals': 7, 'rising pune supergiant': 7, 'rising pune supergiants': 7, 'pune warriors': 7,
    'gt': 8, 'gujarat titans': 8, 'gujarat lions': 8,
    'lsg': 9, 'lucknow super giants': 9,
}
VENUES = {
    'wankhede': 0, 'brabourne': 0, 'dy patil': 0, 'mumbai': 0,
    'chidambaram': 1, 'chepauk': 1, 'chennai': 1,
    'arun jaitley': 2, 'feroz': 2, 'delhi': 2,
    'rajiv': 3, 'uppal': 3, 'hyderabad': 3,
    'eden gardens': 4, 'kolkata': 4,
    'chinnaswamy': 5, 'bangalore': 5, 'bengaluru': 5,
    'sawai': 6, 'jaipur': 6,
    'narendra': 7, 'ahmedabad': 7,
    'ekana': 8, 'lucknow': 8,
}

def normalise_team(name: str) -> int:
    n = str(name).lower().strip()
    return TEAMS.get(n, -1)

def normalise_venue(name: str) -> int:
    n = str(name).lower()
    for key, val in VENUES.items():
        if key in n:
            return val
    return -1

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'  saved -> {path}')

# ── Load data ────────────────────────────────────────────────────────────────
print('Loading data from IPL.csv ...')
ipl_df = pd.read_csv(f'{DATA_DIR}/IPL.csv', low_memory=False)

# Reconstruct matches dataframe
matches = ipl_df.drop_duplicates(subset=['match_id']).copy()
matches = matches.rename(columns={
    'match_id': 'id',
    'match_won_by': 'winner',
})

# Reconstruct team1 and team2 based on first innings
first_rows = ipl_df.drop_duplicates(subset=['match_id']).copy()
matches['team1'] = first_rows['batting_team']
matches['team2'] = first_rows['bowling_team']

# Filter valid matches
if 'result_type' in matches.columns:
    matches = matches[matches['result_type'] != 'no result']

# Normalise season
matches['season'] = matches['season'].astype(str).str[:4].astype(int)
matches = matches[matches['season'] >= MIN_SEASON].copy()

# ── Score data (innings totals) ──────────────────────────────────────────────
deliveries = ipl_df.rename(columns={'innings': 'inning', 'runs_total': 'total_runs'})
inn_totals = deliveries.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
first_inn  = inn_totals[inn_totals['inning'] == 1].rename(columns={'total_runs': 'first_score'})
second_inn = inn_totals[inn_totals['inning'] == 2].rename(columns={'total_runs': 'second_score'})

matches = matches.merge(first_inn[['match_id', 'first_score']], left_on='id', right_on='match_id', how='left')
matches = matches.merge(second_inn[['match_id', 'second_score']], left_on='id', right_on='match_id', how='left')

# Encode basic match info
matches['team1_enc']   = matches['team1'].apply(normalise_team)
matches['team2_enc']   = matches['team2'].apply(normalise_team)
matches['toss_enc']    = matches['toss_winner'].apply(normalise_team)
matches['venue_enc']   = matches['venue'].apply(normalise_venue)
matches['winner_enc']  = matches['winner'].apply(normalise_team)
matches['toss_decision_enc'] = matches['toss_decision'].str.lower().str.strip().apply(lambda x: 1 if x in ('bat', 'batting') else 0)

# Derive batting_first_team and chasing_team
def get_teams(row):
    toss_w = str(row['toss_winner']).lower().strip()
    t1     = str(row['team1']).lower().strip()
    t2     = str(row['team2']).lower().strip()
    decision = str(row['toss_decision']).lower().strip()
    if decision in ('bat', 'batting'):
        batter = toss_w
    else:
        batter = t2 if toss_w == t1 else t1
    chaser = t2 if batter == t1 else t1
    return normalise_team(batter), normalise_team(chaser)

matches[['batting_first_enc', 'chasing_team_enc']] = matches.apply(lambda row: pd.Series(get_teams(row)), axis=1)

base = matches[
    (matches['team1_enc'] != -1) & (matches['team2_enc'] != -1) &
    (matches['toss_enc'] != -1) & (matches['venue_enc'] != -1) &
    (matches['winner_enc'] != -1) & (matches['batting_first_enc'] != -1)
].copy()

print(f'Usable pre-match rows: {len(base)}')

# ── 1. Pre-Match Winner Classifier (XGBoost) ─────────────────────────────────
print('\n[1/4] Training Pre-Match Winner Classifier (XGBoost)...')
X_w = base[['team1_enc', 'team2_enc', 'toss_enc', 'toss_decision_enc', 'venue_enc', 'batting_first_enc', 'chasing_team_enc']]
y_w = base['winner_enc']

X_tr_w, X_te_w, y_tr_w, y_te_w = train_test_split(X_w.values, y_w.values, test_size=0.2, random_state=42)
winner_clf = XGBClassifier(
    n_estimators=300, 
    learning_rate=0.05, 
    max_depth=8, 
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
winner_clf.fit(X_tr_w, y_tr_w)

y_pred_w = winner_clf.predict(X_te_w)
acc_w = accuracy_score(y_te_w, y_pred_w)
print(f'  Accuracy: {acc_w:.3f}')
save(winner_clf, f'{MODEL_DIR}/winner_model.pkl')

# ── 2. Live Win-Probability Classifier (XGBoost) ─────────────────────────────
print('\n[2/4] Building Ball-by-Ball Live Dataset ...')
chase = deliveries[deliveries['inning'] == 2].copy()
chase = chase.merge(first_inn[['match_id', 'first_score']], on='match_id')
chase = chase.merge(base[['id', 'toss_enc', 'toss_decision_enc', 'batting_first_enc', 'chasing_team_enc', 'venue_enc', 'winner_enc']], left_on='match_id', right_on='id', how='inner')

chase['target'] = chase['first_score'] + 1
chase['current_score'] = chase.groupby('match_id')['total_runs'].cumsum()
chase['is_wicket'] = chase['player_out'].notnull().astype(int)
chase['wickets_fallen'] = chase.groupby('match_id')['is_wicket'].cumsum()
chase['wickets_left'] = 10 - chase['wickets_fallen']

chase['is_valid_ball'] = chase['valid_ball'].astype(int)
chase['balls_bowled'] = chase.groupby('match_id')['is_valid_ball'].cumsum()
chase['balls_left'] = 120 - chase['balls_bowled']
chase['balls_left'] = chase['balls_left'].clip(lower=0)

chase['runs_left'] = chase['target'] - chase['current_score']
chase['current_run_rate'] = (chase['current_score'] * 6) / chase['balls_bowled'].clip(lower=1)
chase['required_run_rate'] = (chase['runs_left'] * 6) / chase['balls_left'].clip(lower=1)

# Target variable for live win prob: Did chasing team win? (1 if yes, 0 if no)
chase['chasing_team_won'] = (chase['winner_enc'] == chase['chasing_team_enc']).astype(int)

# Filter out rows with negative runs_left or balls_left=0 to avoid edge case noise
chase = chase[(chase['balls_left'] > 0) & (chase['runs_left'] > 0)]

print(f'Live Dataset size: {len(chase)} balls')
print('\nTraining Live Win-Probability Classifier (XGBoost)...')
live_features = ['toss_enc', 'toss_decision_enc', 'batting_first_enc', 'chasing_team_enc', 'venue_enc', 
                 'target', 'current_score', 'runs_left', 'balls_left', 'wickets_left', 'current_run_rate', 'required_run_rate']

X_live = chase[live_features]
y_live = chase['chasing_team_won']

X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(X_live.values, y_live.values, test_size=0.2, random_state=42)

live_prob_clf = XGBClassifier(
    n_estimators=300, 
    learning_rate=0.05, 
    max_depth=8, 
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
live_prob_clf.fit(X_tr_l, y_tr_l)

y_pred_l = live_prob_clf.predict(X_te_l)
acc_l = accuracy_score(y_te_l, y_pred_l)
prec_l = precision_score(y_te_l, y_pred_l)
rec_l = recall_score(y_te_l, y_pred_l)
cm_l = confusion_matrix(y_te_l, y_pred_l)

print(f'  Accuracy:  {acc_l:.3f}')
print(f'  Precision: {prec_l:.3f}')
print(f'  Recall:    {rec_l:.3f}')
print(f'  Confusion Matrix:\n{cm_l}')
save(live_prob_clf, f'{MODEL_DIR}/win_prob_model.pkl')

# ── 3. First-innings score regressor ─────────────────────────────────────────
print('\n[3/4] Training First-Innings Score Regressor ...')
score_base = base.dropna(subset=['first_score'])
X_s = score_base[['team1_enc', 'team2_enc', 'venue_enc']]
y_s = score_base['first_score']

X_tr, X_te, y_tr, y_te = train_test_split(X_s.values, y_s.values, test_size=0.2, random_state=42)
score_reg1 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
score_reg1.fit(X_tr, y_tr)
mae = mean_absolute_error(y_te, score_reg1.predict(X_te))
print(f'  MAE: {mae:.1f} runs')
save(score_reg1, f'{MODEL_DIR}/score_model_first.pkl')

# ── 4. Second-innings score regressor ────────────────────────────────────────
print('\n[4/4] Training Second-Innings Score Regressor ...')
score_base2 = base.dropna(subset=['second_score'])
X_s2 = score_base2[['team1_enc', 'team2_enc', 'venue_enc', 'first_score']]
y_s2 = score_base2['second_score']

X_tr, X_te, y_tr, y_te = train_test_split(X_s2.values, y_s2.values, test_size=0.2, random_state=42)
score_reg2 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
score_reg2.fit(X_tr, y_tr)
mae2 = mean_absolute_error(y_te, score_reg2.predict(X_te))
print(f'  MAE: {mae2:.1f} runs')
save(score_reg2, f'{MODEL_DIR}/score_model_second.pkl')

print('\nAll models trained and saved to models/')