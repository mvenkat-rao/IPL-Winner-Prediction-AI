"""
Train RandomForest + XGBoost ensemble for winner prediction.
Saves a calibrated stacked classifier to `models/winner_model.pkl` compatible
with the existing `ml_service._load` usage (pickle.load).

Usage:
    python train_models.py --data dataset/ipl_matches_data.csv --out models/

Requires: pandas, numpy, scikit-learn, xgboost
"""

import os
import argparse
import pickle
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, classification_report

from data_service import TEAMS, STADIUMS


def resolve_stadium(name):
    if not isinstance(name, str):
        return 0
    n = name.lower().strip()
    if n in STADIUMS:
        return STADIUMS[n]
    for key, val in STADIUMS.items():
        if key in n or n in key:
            return val
    return 0


def build_features(df):
    # Map team names to integer ids defined in data_service.TEAMS
    def map_team(x):
        # Accept both string team names and numeric encoded team ids
        if pd.isna(x):
            return -1
        if isinstance(x, str):
            k = x.strip().lower()
            return TEAMS.get(k, -1)
        try:
            xi = int(x)
            # If it's already an encoded team id present in TEAMS values, accept it
            if xi in set(TEAMS.values()):
                return xi
        except Exception:
            pass
        return -1

    X = pd.DataFrame()
    X['team1_enc'] = df['team1'].apply(map_team)
    X['team2_enc'] = df['team2'].apply(map_team)
    X['toss_winner_enc'] = df['toss_winner'].apply(lambda x: map_team(x) if isinstance(x, str) else -1)
    X['toss_decision_enc'] = df['toss_decision'].apply(lambda d: 1 if str(d).strip().lower() == 'bat' else 0)
    X['venue_enc'] = df['venue'].apply(resolve_stadium)
    # season numeric (if available)
    X['season'] = pd.to_numeric(df.get('season', df.get('season_id', pd.Series([0]*len(df)))), errors='coerce').fillna(0).astype(int)

    # Simple interaction features to help models
    X['team_diff'] = X['team1_enc'] - X['team2_enc']
    X['toss_is_team1'] = (X['toss_winner_enc'] == X['team1_enc']).astype(int)

    return X


def load_and_prepare(data_path):
    df = pd.read_csv(data_path)

    # Keep only decisive matches where 'result' indicates win and match_winner present
    if 'result' in df.columns:
        df = df[df['result'].str.lower().fillna('') == 'win']

    # Ensure match_winner column exists and map to team id
    if 'match_winner' not in df.columns and 'match_winner' not in df.columns:
        if 'match_winner' not in df.columns:
            raise ValueError('Input CSV must contain a match_winner column with the winner team name or id')

    # Some datasets use 'match_winner' or 'match_won_by' — try both
    if 'match_winner' in df.columns:
        winner_col = 'match_winner'
    elif 'match_won_by' in df.columns:
        winner_col = 'match_won_by'
    else:
        winner_col = 'match_winner'

    # Map winner name to TEAMS id
    def map_winner(x):
        if pd.isna(x):
            return -1
        if isinstance(x, str):
            return TEAMS.get(x.strip().lower(), -1)
        try:
            xi = int(x)
            if xi in set(TEAMS.values()):
                return xi
        except Exception:
            pass
        return -1

    df['winner_enc'] = df[winner_col].apply(map_winner)

    # Drop rows with unknown teams
    def non_empty_team(x):
        if pd.isna(x):
            return False
        if isinstance(x, str):
            return x.strip() != ''
        # numeric values are considered valid (they may be encoded ids)
        return True

    valid_mask = (
        df['team1'].notna() & df['team2'].notna() & df['team1'].apply(non_empty_team) & df['team2'].apply(non_empty_team)
    )
    df = df[valid_mask].copy()

    X = build_features(df)
    y = df['winner_enc'].astype(int)

    # Filter rows with valid labels (present in TEAMS values)
    valid_labels = set(TEAMS.values())
    mask = y.isin(valid_labels)
    X = X[mask]
    y = y[mask]

    return X, y


def train_and_save(X_train, y_train, X_test, y_test, out_dir):
    # Base learners
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='mlogloss', verbosity=0, random_state=42)
        estimators = [('rf', rf), ('xgb', xgb)]
    except Exception:
        print('xgboost not available — training RandomForest only')
        estimators = [('rf', rf)]

    # Stacking
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000),
        n_jobs=-1,
        passthrough=False
    )

    print('Training stacked classifier...')
    stack.fit(X_train, y_train)

    # Calibrate probabilities for better reliability
    print('Calibrating classifier (sigmoid)...')
    cal = CalibratedClassifierCV(stack, method='sigmoid', cv=3)
    cal.fit(X_train, y_train)

    # Evaluate
    preds = cal.predict(X_test)
    probs = cal.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)

    print('\nEvaluation on test set:')
    print('Accuracy:', acc)
    print('Log-loss:', ll)
    print('\nClassification report:')
    print(classification_report(y_test, preds))

    # Save calibrated model using pickle so ml_service._load can load it
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'winner_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(cal, f)
    print(f'Saved calibrated ensemble to {model_path}')

    # Optionally save the raw stack and base models
    with open(os.path.join(out_dir, 'stacking_raw.pkl'), 'wb') as f:
        pickle.dump(stack, f)

    return cal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset/ipl_matches_data.csv', help='Path to matches CSV')
    parser.add_argument('--out', default='models', help='Output directory for models')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    print('Loading dataset:', args.data)
    X, y = load_and_prepare(args.data)
    print('Rows after filtering:', len(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    model = train_and_save(X_train, y_train, X_test, y_test, args.out)


if __name__ == '__main__':
    main()
