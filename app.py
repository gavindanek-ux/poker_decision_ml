"""
Flask web app for the poker coaching model.

Run:
    python app.py

Then open http://localhost:5000 in your browser.
Requires model.pkl and feature_columns.pkl — run train.py first.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_PATH = Path("model.pkl")
COLS_PATH = Path("feature_columns.pkl")

try:
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(COLS_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    model = None
    feature_cols = []
    print("WARNING: model.pkl not found. Run train.py first.")

ACTION_LABELS = {0: "Fold", 1: "Call", 2: "Raise"}
STREET_INT = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

# ---------------------------------------------------------------------------
# Hand strength helpers
# ---------------------------------------------------------------------------

def chen_formula(hole_cards: list[str]) -> float:
    """Normalised Chen formula for preflop hand strength."""
    if len(hole_cards) != 2:
        return 0.5
    try:
        r1 = "23456789TJQKA".index(hole_cards[0][0].upper())
        r2 = "23456789TJQKA".index(hole_cards[1][0].upper())
        suited = hole_cards[0][1].lower() == hole_cards[1][1].lower()
        hi, lo = max(r1, r2), min(r1, r2)
        chen_ranks = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10]
        score = chen_ranks[hi]
        if hi == lo:
            score = max(score * 2, 5)
        else:
            if suited:
                score += 2
            gap = hi - lo - 1
            score -= [0, 1, 2, 4, 5][min(gap, 4)]
            if hi < 12 and gap <= 2:
                score += 1
        return max(0.0, min(1.0, (score + 1) / 21.0))
    except (ValueError, IndexError):
        return 0.5


def compute_hand_strength(hole_cards: list[str], board_cards: list[str]) -> float:
    if len(board_cards) < 3:
        return chen_formula(hole_cards)
    try:
        from treys import Card, Evaluator
        evaluator = Evaluator()
        hand = [Card.new(c) for c in hole_cards]
        board = [Card.new(c) for c in board_cards]
        score = evaluator.evaluate(board, hand)
        return 1.0 - (score / 7462.0)
    except Exception:
        return chen_formula(hole_cards)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_features(data: dict) -> dict:
    hole_cards = data.get("hole_cards", [])
    board_cards = data.get("board_cards", [])
    street = data.get("street", "preflop")
    pot_size = float(data.get("pot_size", 100))
    bet_to_call = float(data.get("bet_to_call", 0))
    position = int(data.get("position", 0))
    num_players = int(data.get("num_players", 6))

    hand_strength = compute_hand_strength(hole_cards, board_cards)

    # Pot odds: fraction of total pot you need to win to break even
    total_pot = pot_size + bet_to_call
    pot_odds = bet_to_call / total_pot if total_pot > 0 else 0.0

    return {
        "hand_strength": hand_strength,
        "pot_odds": pot_odds,
        "position": position,
        "num_players": num_players,
        "street": STREET_INT.get(street, 0),
        "is_last_to_act": int(position == num_players - 1),
        # extras for explanation (not fed to model)
        "_hand_strength_pct": round(hand_strength * 100, 1),
        "_pot_odds_pct": round(pot_odds * 100, 1),
        "_position_label": _position_label(position, num_players),
    }


def _position_label(position: int, num_players: int) -> str:
    thirds = num_players / 3
    if position < thirds:
        return "early"
    if position < 2 * thirds:
        return "middle"
    return "late"


# ---------------------------------------------------------------------------
# Explanation generation
# ---------------------------------------------------------------------------

def generate_explanation(
    prediction: int,
    features: dict,
    probas: list[float],
) -> str:
    hs = features["hand_strength"]
    po = features["pot_odds"]
    pos_label = features["_position_label"]
    hs_pct = features["_hand_strength_pct"]
    po_pct = features["_pot_odds_pct"]
    confidence = round(max(probas) * 100, 1)

    parts = []

    if prediction == 0:  # fold
        if po > 0 and hs < po:
            parts.append(
                f"Your hand equity (~{hs_pct}%) is below the {po_pct}% "
                f"needed to call profitably."
            )
        elif hs < 0.35:
            parts.append(f"With only ~{hs_pct}% hand strength, this hand is too weak to continue.")
        else:
            parts.append("The situation does not justify continuing with this hand.")
        parts.append("Folding minimises losses in a -EV spot.")

    elif prediction == 1:  # call
        if po > 0:
            parts.append(
                f"Your hand equity (~{hs_pct}%) exceeds the pot odds requirement of {po_pct}%, "
                f"making a call profitable long-term."
            )
        else:
            parts.append(f"With ~{hs_pct}% hand strength, calling is justified here.")
        if pos_label == "late":
            parts.append("Acting in late position gives you more information before future streets.")

    elif prediction == 2:  # raise
        parts.append(
            f"Strong equity (~{hs_pct}%) supports building the pot with a raise."
        )
        if pos_label == "late":
            parts.append(
                "Raising in late position forces opponents to act first on future streets, "
                "maximising your positional advantage."
            )
        elif pos_label == "early":
            parts.append(
                "Raising from early position represents a strong range and can take the pot "
                "uncontested or charge drawing hands."
            )

    if confidence < 55:
        parts.append(
            f"This is a close spot (model confidence: {confidence}%) — "
            "consider stack depths and opponent tendencies carefully."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    data = request.json or {}

    try:
        features = build_features(data)
    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    X = pd.DataFrame([{k: features[k] for k in feature_cols}])
    prediction = int(model.predict(X)[0])
    probas = model.predict_proba(X)[0].tolist()

    explanation = generate_explanation(prediction, features, probas)

    return jsonify({
        "action": ACTION_LABELS[prediction],
        "confidence": round(max(probas) * 100, 1),
        "probabilities": {
            "fold": round(probas[0] * 100, 1),
            "call": round(probas[1] * 100, 1),
            "raise": round(probas[2] * 100, 1),
        },
        "explanation": explanation,
        "hand_strength_pct": features["_hand_strength_pct"],
        "pot_odds_pct": features["_pot_odds_pct"],
    })


if __name__ == "__main__":
    app.run(debug=True)
