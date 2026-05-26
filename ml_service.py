import os
import pickle
import random
import numpy as np
import pandas as pd
import traceback

from data_service import (
    TEAMS, TEAM_NAMES, STADIUMS, 
    STADIUM_AVG_SCORE, TEAM_AVG_SCORE, TEAM_WIN_RATE
)

# ── Load models (graceful fallback if not yet trained) ──────────────────────
def _load(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

winner_model       = _load('models/winner_model.pkl')
score_model_first  = _load('models/score_model_first.pkl')
score_model_second = _load('models/score_model_second.pkl')
win_prob_model     = _load('models/win_prob_model.pkl')
live_lgbm_model    = _load('models/live_lgbm_model.pkl')

def _resolve_stadium(name):
    """Resolve a stadium name to its encoded value, trying exact match then keyword."""
    n = name.lower().strip()
    if n in STADIUMS:
        return STADIUMS[n]
    # Keyword fallback: match any key that contains part of the input or vice versa
    for key, val in STADIUMS.items():
        if key in n or n in key:
            return val
    return 0  # default fallback


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID PROBABILITY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _compress_probability(raw_p):
    """Gently compress extreme probabilities toward 50% while keeping the spread realistic.
    Maps: 90%→82%, 80%→74%, 70%→66%, 60%→58%, 50%→50%
    """
    return 0.5 + ((raw_p - 0.5) * 0.80)


def _dynamic_match_day_factor():
    """Random match-day factor simulating dew, pitch, pressure, form, etc."""
    return random.uniform(-0.08, 0.08)


def _phase_of_chase(balls_left):
    if balls_left >= 96:
        return 'powerplay'
    if balls_left > 36:
        return 'middle'
    return 'death'


def _normalize_feature(value, low, high):
    return float(np.clip((value - low) / max(1e-6, high - low), 0.0, 1.0))


def _soft_calibrate_probability(prob):
    # Avoid impossible peaks while preserving strong momentum swings.
    adjusted = 0.5 + np.tanh((prob - 0.5) * 2.0) / 2.0
    return float(np.clip(adjusted, 0.05, 0.92))


def _apply_cricket_rules(prob, runs_left, balls_left, wickets_left, rrr,
                         dot_ball_pct, recent_wickets, boundaries_last_12):
    rule_adj = 0.0
    if wickets_left <= 2 and rrr > 12:
        rule_adj -= 0.22
    if balls_left < 12 and runs_left > 30:
        rule_adj -= 0.20
    if balls_left <= 36 and rrr > 8 and wickets_left <= 6:
        rule_adj -= min(0.22, (rrr - 8) * 0.04 * max(1, 7 - wickets_left))
    if balls_left <= 24 and runs_left > 30 and wickets_left <= 6:
        rule_adj -= 0.10
    if wickets_left >= 7 and rrr < 8:
        rule_adj += 0.14
    if boundaries_last_12 >= 3:
        rule_adj += 0.10
    if recent_wickets >= 2:
        rule_adj -= 0.12
    if dot_ball_pct > 50:
        rule_adj -= min(0.12, (dot_ball_pct - 50) * 0.002)
    return prob + rule_adj


def _blend_model_probs(heuristic_prob, model_probs):
    if not model_probs:
        return heuristic_prob
    model_probs = [float(p) for p in model_probs if p is not None]
    if not model_probs:
        return heuristic_prob
    weight = 0.6 if len(model_probs) == 1 else 0.5
    ml_avg = sum(model_probs) / len(model_probs)
    return float(np.clip((heuristic_prob * (1 - weight)) + (ml_avg * weight), 0.05, 0.92))


def _compute_live_pressure(rrr, wickets_left, dot_ball_pct, recent_wickets,
                           boundaries_last_12, phase):
    pressure = (rrr * 10.0) / max(wickets_left, 1)
    pressure += dot_ball_pct * 0.18
    pressure += max(0, recent_wickets - 1) * 4.0
    pressure -= min(boundaries_last_12, 4) * 3.0

    if phase == 'death':
        pressure += 6.0
    elif phase == 'powerplay':
        pressure -= 3.0

    return float(np.clip(pressure, 0.0, 100.0))


def _momentum_from_metrics(crr, rrr, wickets_left, phase,
                           boundaries_last_12, recent_wickets):
    momentum = 0.0
    momentum += (crr - rrr) * 0.02
    momentum += (wickets_left - 5) * 0.02
    momentum += min(boundaries_last_12, 4) * 0.03
    momentum -= max(0, recent_wickets - 1) * 0.06

    if phase == 'powerplay' and wickets_left >= 8:
        momentum += 0.05
    if phase == 'death' and rrr > 10:
        momentum -= 0.08
    if phase == 'middle' and rrr < 7 and wickets_left >= 7:
        momentum += 0.04

    return float(np.clip(momentum, -0.18, 0.18))


def _venue_chase_bias(batting_team, venue):
    return (TEAM_WIN_RATE.get(batting_team.lower(), 0.52) - 0.50) * 0.2


def _predict_using_models(feat_vec):
    probs = []
    if live_lgbm_model is not None:
        try:
            p = live_lgbm_model.predict_proba([feat_vec])[0][1]
            probs.append(float(p))
        except Exception:
            pass
    if win_prob_model is not None:
        try:
            p = win_prob_model.predict_proba([feat_vec])[0][1]
            probs.append(float(p))
        except Exception:
            pass
    return probs


def _safe_round_probability(value):
    return float(round(np.clip(value * 100.0, 0.0, 100.0), 1))


def _compute_simulation_metrics(prob_winner, prob_loser, winner_name, loser_name,
                                 is_upset, first_score, second_score, batting_first, chasing_team):
    """Compute volatility, upset_chance label, momentum_swing, and AI commentary."""
    
    margin = abs(first_score - second_score)
    prob_diff = abs(prob_winner - prob_loser)
    
    # ── Volatility (0–100) ─────────────────────────────────────────────────
    # Close match + close probs = high volatility
    margin_vol = max(0, 100 - margin * 3)            # large margin → low volatility
    prob_vol   = max(0, 100 - prob_diff * 200)        # large prob gap → low volatility
    random_vol = random.uniform(-8, 8)
    volatility = round(min(100, max(5, (margin_vol * 0.5 + prob_vol * 0.5) + random_vol)))
    
    # ── Upset Chance label ────────────────────────────────────────────────
    loser_pct = prob_loser * 100
    if loser_pct >= 40:
        upset_label = 'HIGH'
    elif loser_pct >= 28:
        upset_label = 'MODERATE'
    else:
        upset_label = 'LOW'
    
    # ── Momentum Swing ────────────────────────────────────────────────────
    if is_upset:
        momentum = 'DRAMATIC'
    elif margin <= 8:
        momentum = random.choice(['DRAMATIC', 'STEADY'])
    elif margin <= 25:
        momentum = 'STEADY'
    else:
        momentum = 'CALM'
    
    # ── AI Commentary ─────────────────────────────────────────────────────
    commentary = _generate_commentary(
        winner_name, loser_name, is_upset, margin, first_score, second_score,
        batting_first, chasing_team, prob_winner, momentum
    )
    
    return volatility, upset_label, momentum, commentary


def _generate_commentary(winner, loser, is_upset, margin, first, second,
                          bat_first, chase_team, prob_winner, momentum):
    """Generate dynamic AI match narrative based on simulation outcome."""
    
    lines = []
    
    # Opening
    if is_upset:
        openers = [
            f"🔥 Unexpected turnaround! {winner} pulls off a stunning upset against {loser}!",
            f"⚡ Underdog comeback detected! {winner} defies the odds!",
            f"💥 Shock result! {winner} stuns {loser} in a match nobody saw coming!",
            f"🌪️ Against all predictions, {winner} emerges victorious!",
        ]
    elif margin <= 6:
        openers = [
            f"🏏 Nail-biter! {winner} edges past {loser} in a thriller!",
            f"😮 What a finish! {winner} scrapes through by the narrowest of margins!",
            f"🎯 Heart-stopping climax as {winner} just about holds on!",
        ]
    elif margin <= 20:
        openers = [
            f"✅ Clinical performance by {winner} to secure a comfortable win.",
            f"🏆 {winner} shows championship quality to beat {loser}.",
            f"👊 Solid display from {winner} — {loser} couldn't find the momentum.",
        ]
    else:
        openers = [
            f"💪 Dominant display! {winner} cruises to a comprehensive victory over {loser}.",
            f"🚀 {winner} completely outclasses {loser} in a one-sided affair.",
            f"🏟️ {winner} puts on a masterclass — {loser} never in the contest.",
        ]
    lines.append(random.choice(openers))
    
    # Innings narrative
    winner_low = winner.lower()
    if winner_low == bat_first.lower():
        # Winner batted first
        if first >= 180:
            lines.append(f"A commanding first-innings total of {first} proved too much to chase.")
        elif first <= 140:
            lines.append(f"Despite a below-par {first}, {winner}'s bowling attack defended it superbly.")
        else:
            lines.append(f"{winner} posted a competitive {first} and defended it well.")
    else:
        # Winner chased
        if margin <= 4:
            lines.append(f"A nerve-wracking chase — {winner} reached the target of {first+1} with just {margin} runs to spare!")
        elif second >= 180:
            lines.append(f"An explosive chase of {second} saw {winner} race past the target with authority.")
        else:
            lines.append(f"{winner} chased down {first+1} runs with composure.")
    
    # Momentum flavor
    if momentum == 'DRAMATIC':
        lines.append("Momentum shifted dramatically throughout the match — a true IPL classic!")
    elif momentum == 'STEADY':
        lines.append("The momentum gradually built in the winner's favor as the match progressed.")
    
    # Probability insight
    win_pct = round(prob_winner * 100)
    if is_upset:
        lines.append(f"Pre-match AI gave {winner} only a {win_pct}% chance — proving cricket is gloriously unpredictable!")
    
    return " ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT WINNER (Hybrid Engine)
# ─────────────────────────────────────────────────────────────────────────────

def predict_winner(team1, team2, toss_winner, venue, batting_first=None, toss_decision='bat'):
    """Return predicted winner + win-probability dict + simulation metrics."""
    t1, t2 = team1.lower(), team2.lower()
    tw, v  = toss_winner.lower(), venue.lower()

    if batting_first is None:
        batting_first = toss_winner
    bf = batting_first.lower()

    toss_dec_enc  = 1 if toss_decision == 'bat' else 0
    # For training compatibility we include season and engineered features
    season_enc = 0

    # team encodings
    t1_code = TEAMS.get(t1)
    t2_code = TEAMS.get(t2)
    toss_code = TEAMS.get(tw)
    if t1_code is None or t2_code is None or toss_code is None:
        # Fall back gracefully if the team alias is not recognized.
        t1_enc = TEAMS.get(t1, 0)
        t2_enc = TEAMS.get(t2, 0)
        toss_enc = TEAMS.get(tw, 0)
    else:
        t1_enc = t1_code
        t2_enc = t2_code
        toss_enc = toss_code

    # engineered
    team_diff = t1_enc - t2_enc
    toss_is_team1 = 1 if tw == t1 else 0

    venue_enc = _resolve_stadium(v)
    feature_columns = [
        'team1_enc', 'team2_enc', 'toss_winner_enc', 'toss_decision_enc',
        'venue_enc', 'season', 'team_diff', 'toss_is_team1'
    ]
    features_7 = pd.DataFrame([
        [t1_enc, t2_enc, toss_enc, toss_dec_enc, venue_enc, season_enc, team_diff, toss_is_team1]
    ], columns=feature_columns)

    # ── Historical baseline probabilities ─────────────────────────────────
    bat_bonus = 0.04 if toss_decision == 'bat' else -0.02
    r1 = TEAM_WIN_RATE.get(t1, 0.5) + (0.05 if tw == t1 else 0) + (bat_bonus if bf == t1 else -bat_bonus)
    r2 = TEAM_WIN_RATE.get(t2, 0.5) + (0.05 if tw == t2 else 0) + (bat_bonus if bf == t2 else -bat_bonus)
    total_r = r1 + r2
    hist_p1 = r1 / total_r
    hist_p2 = r2 / total_r

    try:
        if winner_model:
            probs = winner_model.predict_proba(features_7)[0]
            
            raw_probs = {c: float(p) for c, p in zip(winner_model.classes_, probs)
                         if c in (t1_enc, t2_enc)}
            total_p = sum(raw_probs.values())
            
            if total_p > 0:
                ml_p1 = raw_probs.get(TEAMS[t1], 0) / total_p
                ml_p2 = raw_probs.get(TEAMS[t2], 0) / total_p
            else:
                ml_p1, ml_p2 = 0.5, 0.5

            # ── HYBRID ENGINE: 70% ML + 20% Historical + 10% Match-Day ──
            # Apply dynamic match-day factor as a small zero-centered offset
            # so it can swing either team's probability slightly each simulation.
            mdf = _dynamic_match_day_factor()  # in [-0.08, 0.08]

            raw_p1 = (ml_p1 * 0.70) + (hist_p1 * 0.20) + (mdf * 0.10)
            raw_p2 = (ml_p2 * 0.70) + (hist_p2 * 0.20) - (mdf * 0.10)
            
            # Normalize to sum to 1 and guard against extreme values
            total_raw = max(1e-6, raw_p1 + raw_p2)
            raw_p1 = max(0.0001, raw_p1 / total_raw)
            raw_p2 = max(0.0001, raw_p2 / total_raw)
            
            # ── PROBABILITY COMPRESSION ──────────────────────────────────
            comp_p1 = _compress_probability(raw_p1)
            comp_p2 = _compress_probability(raw_p2)
            
            # Re-normalize after compression
            total_comp = max(1e-6, comp_p1 + comp_p2)
            final_p1 = comp_p1 / total_comp
            final_p2 = comp_p2 / total_comp

            # Keep probabilities within the 45-55 band, but allow natural fluctuation.
            if final_p1 < 0.45:
                final_p1 = 0.45 + random.uniform(0.0, 0.05)
                final_p2 = 1.0 - final_p1
            elif final_p1 > 0.55:
                final_p1 = 0.55 - random.uniform(0.0, 0.05)
                final_p2 = 1.0 - final_p1

            t1_name = TEAM_NAMES[TEAMS[t1]]
            t2_name = TEAM_NAMES[TEAMS[t2]]
            
            prob_dict = {
                t1_name: round(final_p1 * 100, 1),
                t2_name: round(final_p2 * 100, 1)
            }
            
            # ── WEIGHTED RANDOM SIMULATION ───────────────────────────────
            # Slightly sharpen the weight gap so favorites still win significantly more often,
            # while allowing true underdog upsets when probability gaps are moderate.
            power = 1.12
            weighted_p1 = max(final_p1, 1e-6) ** power
            weighted_p2 = max(final_p2, 1e-6) ** power
            winner = random.choices(
                [t1_name, t2_name],
                weights=[weighted_p1, weighted_p2],
                k=1
            )[0]
            
            # Determine if this is an upset (underdog won)
            is_upset = (winner == t1_name and final_p1 < final_p2) or \
                       (winner == t2_name and final_p2 < final_p1)
            
            # Also return the underlying final probabilities so callers can compute
            # numeric upset chance for the losing side.
            return winner, prob_dict, final_p1, final_p2, is_upset
            
    except Exception as e:
        print(f"Error predicting winner with XGBoost: {e}")
        traceback.print_exc()

    # ── Fallback heuristic ────────────────────────────────────────────────
    # Fallback heuristic: use historical baseline plus a small dynamic factor
    mdf = _dynamic_match_day_factor()
    raw_p1 = hist_p1 + (mdf * 0.10)
    raw_p2 = hist_p2 - (mdf * 0.10)
    total_f = raw_p1 + raw_p2
    raw_p1 /= total_f
    raw_p2 /= total_f
    
    comp_p1 = _compress_probability(raw_p1)
    comp_p2 = _compress_probability(raw_p2)
    total_comp = comp_p1 + comp_p2
    final_p1 = comp_p1 / total_comp
    final_p2 = comp_p2 / total_comp

    if final_p1 < 0.45:
        final_p1 = 0.45 + random.uniform(0.0, 0.05)
        final_p2 = 1.0 - final_p1
    elif final_p1 > 0.55:
        final_p1 = 0.55 - random.uniform(0.0, 0.05)
        final_p2 = 1.0 - final_p1
    
    winner = random.choices(
        [t1.upper(), t2.upper()],
        weights=[final_p1, final_p2],
        k=1
    )[0]
    
    is_upset = (winner == t1.upper() and final_p1 < final_p2) or \
               (winner == t2.upper() and final_p2 < final_p1)
    
    prob_dict = {t1.upper(): round(final_p1 * 100, 1), t2.upper(): round(final_p2 * 100, 1)}
    return winner, prob_dict, final_p1, final_p2, is_upset


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT SCORES (Advanced Score Engine)
# ─────────────────────────────────────────────────────────────────────────────

def predict_scores(team1, team2, venue, simulated_winner=None):
    """Predict 1st and 2nd innings totals with realistic variance, collapses, and clutch finishes."""
    t1, t2, v = team1.lower(), team2.lower(), venue.lower()
    venue_enc = _resolve_stadium(v)
    features = [[TEAMS[t1], TEAMS[t2], venue_enc]]

    first, second = 160, 150
    try:
        if score_model_first:
            base_first  = int(score_model_first.predict(features)[0])
            
            features2 = [[TEAMS[t1], TEAMS[t2], venue_enc, base_first]]
            base_second = int(score_model_second.predict(features2)[0]) if score_model_second else base_first - 8
            
            # ── Adaptive variance (±10 to ±20 runs) ─────────────────────
            variance_range = random.randint(10, 20)
            first  = base_first  + random.randint(-variance_range, variance_range)
            second = base_second + random.randint(-variance_range, variance_range)
            
            # ── Simulate batting collapse (12% chance) ───────────────────
            if random.random() < 0.12:
                collapse_target = random.choice(['first', 'second'])
                collapse_amount = random.randint(25, 55)
                if collapse_target == 'first':
                    first -= collapse_amount
                else:
                    second -= collapse_amount
                    
        else:
            base   = (STADIUM_AVG_SCORE.get(v, 160) + TEAM_AVG_SCORE.get(t1, 160)) / 2
            variance_range = random.randint(10, 20)
            first  = int(base + random.randint(-variance_range, variance_range))
            second = int(first + random.randint(-variance_range, variance_range))
    except Exception as e:
        print(f"Error predicting scores: {e}")
        traceback.print_exc()
        base   = (STADIUM_AVG_SCORE.get(v, 160) + TEAM_AVG_SCORE.get(t1, 160)) / 2
        first  = int(base + random.randint(-15, 16))
        second = int(first + random.randint(-15, 16))

    # Ensure scores stay in a realistic T20 range
    first  = max(100, min(260, first))
    second = max(90, min(250, second))

    # ── Adjust scores to match simulated winner ──────────────────────────
    if simulated_winner:
        sim_w_lower = simulated_winner.lower()
        if sim_w_lower == t1:  # Batting-first team wins
            if second >= first:
                # Random deficit for the chaser
                second = first - random.randint(5, 36)
        elif sim_w_lower == t2:  # Chasing team wins
            if second < first:
                # ── Clutch finish (30% chance for nail-biters) ───────────
                if random.random() < 0.30:
                    second = first + random.choice([1, 2, 3])   # Last-ball drama
                else:
                    second = first + random.randint(1, 24)       # Comfortable-to-tight chase

    # Final sanity check to keep scores in bounds after adjustment
    first  = max(100, min(260, first))
    second = max(90, min(250, second))

    return int(first), int(second)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE WIN PROBABILITY
# ─────────────────────────────────────────────────────────────────────────────

def live_win_probability(batting_team, bowling_team, target,
                         current_score, wickets, balls_bowled,
                         venue='wankhede stadium, mumbai', toss_winner=None, toss_decision='bowl',
                         recent_wickets=0, dot_ball_pct=0.0, boundaries_last_12=0,
                         batting_team_recent_form=0.5, bowling_team_death_overs_strength=0.5):
    """Advanced cricket intelligence live win probability for the batting team."""
    balls_left = max(0, 120 - balls_bowled)
    runs_left  = max(0, target - current_score)
    wickets_left = max(0, 10 - wickets)
    crr = (current_score * 6.0) / max(balls_bowled, 1)
    rrr = (runs_left * 6.0) / max(balls_left, 1) if balls_left > 0 else 999.0

    phase = _phase_of_chase(balls_left)
    pressure_index = _compute_live_pressure(rrr, wickets_left, dot_ball_pct, recent_wickets, boundaries_last_12, phase)
    momentum = _momentum_from_metrics(crr, rrr, wickets_left, phase, boundaries_last_12, recent_wickets)
    venue_bias = _venue_chase_bias(batting_team, venue)

    score_gap = runs_left / max(target, 1)
    base_prob = 0.50
    base_prob += (wickets_left - 5) * 0.03
    base_prob += (crr - rrr) * 0.02
    base_prob += momentum
    base_prob += venue_bias
    base_prob += batting_team_recent_form * 0.05
    base_prob -= bowling_team_death_overs_strength * 0.06
    base_prob -= (pressure_index / 100.0) * 0.18
    base_prob -= score_gap * 0.12

    # Match-phase handling
    if phase == 'powerplay':
        base_prob += 0.04 if rrr <= 8 else -0.05
    elif phase == 'middle':
        base_prob += 0.02 if rrr <= 7 else -0.035
    else:
        base_prob -= 0.06 if rrr > 9 else 0.0

    # Rule-based cricket logic
    adjusted_prob = _apply_cricket_rules(base_prob, runs_left, balls_left, wickets_left, rrr,
                                         dot_ball_pct, recent_wickets, boundaries_last_12)

    # Model ensemble blend
    feat_vec = [
        TEAMS.get((toss_winner or batting_team).lower(), TEAMS.get(batting_team.lower(), 0)),
        1 if toss_decision == 'bat' else 0,
        TEAMS.get(batting_team.lower(), 0),
        TEAMS.get(bowling_team.lower(), 0),
        _resolve_stadium(venue.lower()),
        target, current_score, runs_left, balls_left, wickets_left, crr, rrr,
        recent_wickets, dot_ball_pct, boundaries_last_12,
        batting_team_recent_form, bowling_team_death_overs_strength
    ]

    model_probs = _predict_using_models(feat_vec)
    if model_probs:
        adjusted_prob = _blend_model_probs(adjusted_prob, model_probs)

    calibrated = _soft_calibrate_probability(np.clip(adjusted_prob, 0.05, 0.92))

    if current_score >= target:
        return 100.0
    if wickets_left <= 0 or balls_left <= 0:
        return 0.0

    return _safe_round_probability(calibrated)
