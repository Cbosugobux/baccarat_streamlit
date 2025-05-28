import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the cleaned file
df = pd.read_csv("baccarat_log_cleaned.csv")

# Map outcome to binary: 0 = Player, 2 = Banker
df = df[df["outcome"].isin([0, 2])]
df["outcome"] = df["outcome"].map({0: 0, 2: 1})  # Player=0, Banker=1

# Create shoe_phase_encoded
def classify_shoe_phase(hand_num):
    if hand_num <= 15:
        return 0
    elif hand_num <= 50:
        return 1
    else:
        return 2

df["shoe_phase_encoded"] = df["hand"].apply(classify_shoe_phase)

# Define features and target
X = df[["P1", "B1", "P2", "B2", "shoe_phase_encoded"]]
y = df["outcome"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open("LastChance/baccarat_side_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Side prediction model saved as baccarat_side_model.pkl")
