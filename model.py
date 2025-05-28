# train_baccarat_filter_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Load cleaned and labeled data
df = pd.read_csv("baccarat_log_cleaned.csv")

# Keep only valid outcomes
df = df[df['outcome'].isin([0, 1, 2])]
df = df.dropna(subset=['P1', 'P2', 'B1', 'B2', 'hand'])

# Step 1: Shoe Phase Feature
def classify_shoe_phase(hand_number):
    if hand_number <= 15:
        return "early"
    elif hand_number <= 50:
        return "middle"
    else:
        return "late"

df['shoe_phase'] = df['hand'].astype(int).apply(classify_shoe_phase)
df['shoe_phase_encoded'] = df['shoe_phase'].map({"early": 0, "middle": 1, "late": 2})

# Step 2: Label whether a hand is worth betting on (mid-shoe Banker wins only)
df['should_bet'] = ((df['shoe_phase'] == 'middle') & (df['outcome'] == 2)).astype(int)

# Step 3: Define features and target
X = df[['P1', 'B1', 'P2', 'B2', 'shoe_phase_encoded']]
y = df['should_bet']

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# Step 5: Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Step 7: Save model
with open("LastChance/baccarat_rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as baccarat_rf_model.pkl")
