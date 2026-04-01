"""
Train a poker decision model on parsed IRC hand data.

Usage:
    python train.py                        # train on data/hands.csv
    python train.py --csv data/hands.csv   # explicit path
    python train.py --sample 200000        # use a random sample for speed
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "hand_strength",
    "pot_odds",
    "position",
    "num_players",
    "street",
    "is_last_to_act",
]

ACTION_NAMES = {0: "fold", 1: "call", 2: "raise"}

MODEL_PATH = Path("model.pkl")
COLS_PATH = Path("feature_columns.pkl")


def derive_correct_action(hand_strength: float, pot_odds: float, position: int, num_players: int) -> int:
    """
    Derive the strategically correct action from hand features.
    This replaces the player's actual action as the label, giving us
    balanced fold/call/raise classes for every hand in the dataset.

    Logic based on fundamental poker theory:
    - Fold if equity < pot odds required (calling is -EV)
    - Raise with strong hands, especially in position
    - Call when equity justifies it but hand isn't strong enough to raise
    """
    position_bonus = position / max(num_players, 1) * 0.05  # late position loosens ranges slightly
    adjusted_strength = hand_strength + position_bonus

    # Not enough equity to call — fold
    if adjusted_strength < pot_odds:
        return 0

    # Strong hand: raise
    if adjusted_strength > 0.65:
        return 2

    # Marginal but +EV: call
    return 1


def load_data(csv_path: Path, sample: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path}")

    df = df.dropna(subset=FEATURE_COLS)
    print(f"After dropping nulls: {len(df):,} rows")

    # Derive correct action label from hand features (not what player actually did)
    df["action_label"] = df.apply(
        lambda r: derive_correct_action(
            r["hand_strength"], r["pot_odds"], r["position"], r["num_players"]
        ),
        axis=1,
    )
    print("Relabelled using strategy-derived correct actions.")

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42)
        print(f"Sampled down to {len(df):,} rows")

    return df


def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )),
    ])


def train(csv_path: Path, sample: int | None = None):
    df = load_data(csv_path, sample)

    X = df[FEATURE_COLS].astype(float)
    y = df["action_label"].astype(int)

    print("\nClass distribution:")
    for label, count in y.value_counts().items():
        print(f"  {ACTION_NAMES[label]:6s}: {count:,} ({count/len(y)*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining on {len(X_train):,} samples...")
    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["fold", "call", "raise"]))

    # Feature importances
    importances = model.named_steps["clf"].feature_importances_
    print("Feature importances:")
    for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        print(f"  {feat:20s}: {imp:.4f}")

    # Save
    joblib.dump(model, MODEL_PATH)
    joblib.dump(FEATURE_COLS, COLS_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Feature columns saved to {COLS_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/hands.csv"))
    parser.add_argument("--sample", type=int, default=None,
                        help="Use a random subset for faster training")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found. Run parse_irc.py first.")
        return

    train(args.csv, sample=args.sample)


if __name__ == "__main__":
    main()
