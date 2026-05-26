import json
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'static_data.json')

def load_data():
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

_data = load_data()

TEAMS = _data.get("TEAMS", {})
TEAM_NAMES = {v: k.upper() for k, v in TEAMS.items()}
STADIUMS = {k.lower(): v for k, v in _data.get("STADIUMS", {}).items()}
STADIUM_AVG_SCORE = {k.lower(): v for k, v in _data.get("STADIUM_AVG_SCORE", {}).items()}
TEAM_AVG_SCORE = _data.get("TEAM_AVG_SCORE", {})
TEAM_WIN_RATE = _data.get("TEAM_WIN_RATE", {})
PLAYER_STATS = _data.get("PLAYER_STATS", {})
