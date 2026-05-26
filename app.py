from flask import Flask, request, render_template, jsonify
from data_service import TEAMS, STADIUMS
from ml_service import predict_winner, predict_scores, live_win_probability

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html',
                           teams=sorted(TEAMS.keys()),
                           stadiums=sorted(STADIUMS.keys()))

@app.route('/predict/toss', methods=['POST'])
def predict_toss():
    """Predict toss winner between two teams."""
    import random
    data = request.get_json()
    if not data or 'team1' not in data or 'team2' not in data:
        return jsonify({'error': 'Missing team1 or team2'}), 400
        
    team1 = data['team1']
    team2 = data['team2']
    if team1.lower() == team2.lower():
        return jsonify({'error': 'Team 1 and Team 2 cannot be the same.'}), 400

    winner = random.choice([team1, team2])
    return jsonify({'toss_winner': winner})


@app.route('/predict/match', methods=['POST'])
def predict_match():
    """Pre-match: winner prediction + score forecast + simulation analytics."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON payload provided'}), 400
        
    required = ['team1', 'team2', 'toss', 'venue']
    if not all(k in data for k in required):
        return jsonify({'error': 'Missing required fields for match prediction'}), 400

    team1 = data['team1']
    team2 = data['team2']
    toss_winner = data['toss']
    toss_decision = data.get('toss_decision', 'bat')
    venue = data['venue']

    if team1.lower() == team2.lower():
        return jsonify({'error': 'Team 1 and Team 2 cannot be the same.'}), 400

    if toss_decision == 'bat':
        batting_first = toss_winner
    else:
        batting_first = team2 if toss_winner.lower() == team1.lower() else team1
    chasing_team = team2 if batting_first.lower() == team1.lower() else team1

    winner, probs, final_p1, final_p2, is_upset = predict_winner(
        team1, team2, toss_winner, venue,
        batting_first=batting_first, toss_decision=toss_decision
    )
    first_inn, second_inn = predict_scores(batting_first, chasing_team, venue, simulated_winner=winner)

    return jsonify({
        'winner':                 winner,
        'win_probabilities':      probs,
        'upset_pct':              round((final_p2 if winner.lower() == team1.lower() else final_p1) * 100, 1),
        'simulation_note':        'Weighted simulation used — favorites win more often, but underdogs can still prevail.',
        'predicted_score_first':  int(first_inn),
        'predicted_score_second': int(second_inn),
        'batting_first':          batting_first.upper(),
        'chasing_team':           chasing_team.upper(),
        'toss_decision':          toss_decision,
    })


@app.route('/predict/live', methods=['POST'])
def predict_live():
    """Live chase: dynamic win-probability update."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON payload provided'}), 400
        
    required = ['batting_team', 'bowling_team', 'target', 'current_score', 'wickets', 'balls_bowled']
    if not all(k in data for k in required):
        return jsonify({'error': 'Missing required fields for live prediction'}), 400

    batting = data['batting_team']
    bowling = data['bowling_team']
    if batting.lower() == bowling.lower():
        return jsonify({'error': 'Batting and bowling teams cannot be the same.'}), 400

    try:
        target = int(data['target'])
        current_score = int(data['current_score'])
        wickets = int(data['wickets'])
        balls_bowled = int(data['balls_bowled'])
        recent_wickets = int(data.get('recent_wickets', 0))
        dot_ball_pct = float(data.get('dot_ball_pct', 0))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid numeric data provided'}), 400

    if current_score >= target:
        prob = 100.0
    elif wickets >= 10 or balls_bowled >= 120:
        prob = 0.0
    else:
        prob = live_win_probability(
            batting_team=batting,
            bowling_team=bowling,
            target=target,
            current_score=current_score,
            wickets=wickets,
            balls_bowled=balls_bowled,
            venue=data.get('venue', ''),
            toss_winner=data.get('toss_winner'),
            toss_decision=data.get('toss_decision', 'bowl'),
            recent_wickets=int(data.get('recent_wickets', 0)),
            dot_ball_pct=float(data.get('dot_ball_pct', 0.0)),
            boundaries_last_12=int(data.get('boundaries_last_12', 0)),
            batting_team_recent_form=float(data.get('batting_team_recent_form', 0.5)),
            bowling_team_death_overs_strength=float(data.get('bowling_team_death_overs_strength', 0.5))
        )
        
    batting = batting.upper()
    bowling = bowling.upper()
    
    runs_left = target - current_score
    balls_left = 120 - balls_bowled
    rrr = (runs_left * 6) / max(balls_left, 1) if balls_left > 0 else 0
    
    base_pressure = 30
    rrr_pressure = min(max((rrr - 6) * 5, 0), 40)
    wicket_pressure = wickets * 4 + recent_wickets * 8
    dot_pressure = dot_ball_pct * 0.2
    
    pressure_index = min(max(base_pressure + rrr_pressure + wicket_pressure + dot_pressure, 0), 100)
    
    if pressure_index < 40:
        pressure_level = 'LOW'
        pressure_msg = 'Batting team is cruising comfortably.'
    elif pressure_index < 70:
        pressure_level = 'MEDIUM'
        pressure_msg = 'Game is balanced. One good over can shift momentum.'
    else:
        pressure_level = 'HIGH'
        if rrr > 10:
            pressure_msg = 'Pressure increasing due to high required run rate.'
        elif recent_wickets >= 2:
            pressure_msg = 'Recent wickets have put the batting team under severe pressure.'
        else:
            pressure_msg = 'Bowling team dominating with tight overs and pressure.'
            
    return jsonify({
        'batting_team':      batting,
        'bowling_team':      bowling,
        'batting_win_prob':  prob,
        'bowling_win_prob':  round(100 - prob, 1),
        'pressure_index':    round(pressure_index, 1),
        'pressure_level':    pressure_level,
        'pressure_msg':      pressure_msg
    })


if __name__ == '__main__':
    app.run(debug=True)