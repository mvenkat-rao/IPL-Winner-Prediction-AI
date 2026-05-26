import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from data_service import TEAMS, VENUES

print("--- IPL Predictor Pro: Dynamic Accuracy Tester ---")

def normalise_team(name: str) -> int:
    return TEAMS.get(str(name).lower().strip(), -1)

def normalise_venue(name: str) -> int:
    n = str(name).lower()
    for key, val in VENUES.items():
        if key in n:
            return val
    return -1

def load_model(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

live_prob_model = load_model('models/win_prob_model.pkl')
if not live_prob_model:
    print("Models not found! Please run `python model.py` first to train the models.")
    exit(1)

print("Loading test data from IPL.csv...")
ipl_df = pd.read_csv('dataset/IPL.csv', low_memory=False)

# Reconstruct a small test set (e.g., just the last 100 matches)
matches = ipl_df.drop_duplicates(subset=['match_id']).copy()
matches = matches.rename(columns={'match_id': 'id', 'match_won_by': 'winner'})
first_rows = ipl_df.drop_duplicates(subset=['match_id']).copy()
matches['team1'] = first_rows['batting_team']
matches['team2'] = first_rows['bowling_team']

# Get the most recent 100 matches
matches['season'] = matches['season'].astype(str).str[:4].astype(int)
matches = matches.sort_values(by=['season', 'id'], ascending=False)
test_matches = matches.head(100).copy()

deliveries = ipl_df.rename(columns={'innings': 'inning', 'runs_total': 'total_runs'})
inn_totals = deliveries.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
first_inn  = inn_totals[inn_totals['inning'] == 1].rename(columns={'total_runs': 'first_score'})

test_matches = test_matches.merge(first_inn[['match_id', 'first_score']], left_on='id', right_on='match_id', how='inner')

test_matches['toss_enc']    = test_matches['toss_winner'].apply(normalise_team)
test_matches['venue_enc']   = test_matches['venue'].apply(normalise_venue)
test_matches['winner_enc']  = test_matches['winner'].apply(normalise_team)
test_matches['toss_decision_enc'] = test_matches['toss_decision'].str.lower().str.strip().apply(lambda x: 1 if x in ('bat', 'batting') else 0)

def get_teams(row):
    toss_w = str(row['toss_winner']).lower().strip()
    t1     = str(row['team1']).lower().strip()
    t2     = str(row['team2']).lower().strip()
    decision = str(row['toss_decision']).lower().strip()
    batter = toss_w if decision in ('bat', 'batting') else (t2 if toss_w == t1 else t1)
    chaser = t2 if batter == t1 else t1
    return normalise_team(batter), normalise_team(chaser)

test_matches[['batting_first_enc', 'chasing_team_enc']] = test_matches.apply(lambda row: pd.Series(get_teams(row)), axis=1)

chase = deliveries[deliveries['match_id'].isin(test_matches['id']) & (deliveries['inning'] == 2)].copy()
chase = chase.merge(first_inn[['match_id', 'first_score']], on='match_id')
chase = chase.merge(test_matches[['id', 'toss_enc', 'toss_decision_enc', 'batting_first_enc', 'chasing_team_enc', 'venue_enc', 'winner_enc']], left_on='match_id', right_on='id', how='inner')

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
chase['chasing_team_won'] = (chase['winner_enc'] == chase['chasing_team_enc']).astype(int)
chase = chase[(chase['balls_left'] > 0) & (chase['runs_left'] > 0)]

print(f"\nEvaluating Live Win-Probability Model on Unseen Data (Last 100 matches, {len(chase)} balls)...")

live_features = ['toss_enc', 'toss_decision_enc', 'batting_first_enc', 'chasing_team_enc', 'venue_enc', 
                 'target', 'current_score', 'runs_left', 'balls_left', 'wickets_left', 'current_run_rate', 'required_run_rate']

X_test = chase[live_features]
y_test = chase['chasing_team_won']

predictions = live_prob_model.predict(X_test.values)

print("\n--- Live Model Test Results ---")
print(f"Accuracy:  {accuracy_score(y_test, predictions) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, predictions) * 100:.2f}%")
print(f"Recall:    {recall_score(y_test, predictions) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
