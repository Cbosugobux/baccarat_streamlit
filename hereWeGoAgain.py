# baccarat_next_bet_cli.py

import pickle
import pandas as pd
import os

# Load trained model
MODEL_PATH = "baccarat_rf_model.pkl"
LOG_PATH = "LastChance/baccarat_live_log.csv"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Helper: Classify shoe phase
def classify_shoe_phase(hand_num):
    if hand_num <= 15:
        return 0  # early
    elif hand_num <= 50:
        return 1  # middle
    else:
        return 2  # late

# Main loop
print("\n🎰 Baccarat Live CLI — Predict NEXT Hand Bet")
print("Type 'new' to start a new shoe or 'exit' to quit.\n")

records = []
hand = None

while True:
    try:
        if hand is None:
            start_input = input("Enter starting hand number (or 'exit'): ")
            if start_input.lower() == 'exit':
                break
            hand = int(start_input)
            continue

        cmd = input(f"\n--- Hand #{hand} --- (enter 'new' to reset shoe, 'exit' to quit)\n  Press Enter to input hand, or type command: ").strip().lower()
        if cmd == 'exit':
            break
        elif cmd == 'new':
            hand = None
            continue

        p1 = int(input("  P1: "))
        p2 = int(input("  P2: "))
        b1 = int(input("  B1: "))
        b2 = int(input("  B2: "))
        p3 = input("  Player drew 3rd card? (1=yes, 0=no): ")
        b3 = input("  Banker drew 3rd card? (1=yes, 0=no): ")
        outcome = input("  Outcome of this hand (P=Player, B=Banker, T=Tie): ").strip().upper()

        shoe_phase = classify_shoe_phase(hand)

        # Build input for model — reorder to P1, B1, P2, B2 internally
        model_input = pd.DataFrame([[p1, b1, p2, b2, shoe_phase]], columns=["P1", "B1", "P2", "B2", "shoe_phase_encoded"])
        prediction = model.predict(model_input)[0]
        pred_text = "✅ BET on next hand" if prediction == 1 else "🚫 Do NOT bet next hand"

        print(f"\nPrediction for NEXT hand after hand #{hand}: {pred_text}\n")

        # Log result for later retraining
        record = {
            "hand": hand,
            "P1": p1,
            "P2": p2,
            "B1": b1,
            "B2": b2,
            "P3": p3,
            "B3": b3,
            "outcome": outcome,
            "shoe_phase_encoded": shoe_phase,
            "should_bet_next": prediction
        }
        records.append(record)
        hand += 1

    except Exception as e:
        print(f"❌ Error: {e}\n")

# Save log if any hands were recorded
if records:
    log_df = pd.DataFrame(records)
    if os.path.exists(LOG_PATH):
        log_df.to_csv(LOG_PATH, mode='a', header=False, index=False)
    else:
        log_df.to_csv(LOG_PATH, index=False)

    print(f"\n📝 Saved {len(records)} hands to {LOG_PATH}")
else:
    print("\nNo data to save. Exiting.")
