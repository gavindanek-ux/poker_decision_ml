import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Reproducibility
np.random.seed(42)

# Simulate poker-inspired data
n = 1000
data = pd.DataFrame({
    "hand_strength": np.random.rand(n),         # 0 to 1
    "pot_size": np.random.randint(10, 101, n),  # 10 to 100
    "bet_size": np.random.randint(5, 51, n),    # 5 to 50
    "position": np.random.randint(0, 2, n)      # 0 = early, 1 = late
})

# Simple rule-based labels for action
def decision(row):
    if row["hand_strength"] > 0.75:
        return 2  # raise
    elif row["hand_strength"] > 0.45:
        return 1  # call
    else:
        return 0  # fold

data["action"] = data.apply(decision, axis=1)

# Features and target
X = data[["hand_strength", "pot_size", "bet_size", "position"]]
y = data["action"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["fold", "call", "raise"]))

# Feature importance
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importances:\n")
print(feature_importance)