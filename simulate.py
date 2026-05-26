"""
Run N simulated predictions using the prediction API to measure empirical win rates.
Usage:
    python simulate.py --team1 rcb --team2 mi --toss rcb --venue "M. Chinnaswamy Stadium, Bengaluru" --n 1000
"""
import argparse
from collections import Counter

from ml_service import predict_winner


def run_sim(team1, team2, toss, venue, toss_decision='bat', n=1000):
    counts = Counter()
    prob_sums = {team1.upper(): 0.0, team2.upper(): 0.0}
    for i in range(n):
        winner, probs, p1, p2, is_upset = predict_winner(team1, team2, toss, venue, batting_first=toss if toss_decision=='bat' else team2, toss_decision=toss_decision)
        counts[winner] += 1
        # accumulate displayed probabilities from probs dict
        # ensure keys
        k1 = list(probs.keys())[0]
        k2 = list(probs.keys())[1]
        prob_sums[k1] += probs[k1]
        prob_sums[k2] += probs[k2]
    print(f"Simulated {n} matches: {team1.upper()} vs {team2.upper()} at {venue}")
    for team in [team1.upper(), team2.upper()]:
        wins = counts.get(team, 0)
        avg_prob = prob_sums.get(team, 0) / n
        print(f"{team}: wins={wins} ({wins/n*100:.2f}%), avg_displayed_prob={avg_prob:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--team1', default='rcb')
    parser.add_argument('--team2', default='mi')
    parser.add_argument('--toss', default='rcb')
    parser.add_argument('--venue', default='M. Chinnaswamy Stadium, Bengaluru')
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()
    run_sim(args.team1, args.team2, args.toss, args.venue, toss_decision='bat', n=args.n)
