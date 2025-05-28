import pandas as pd

# Load your messy file
df = pd.read_csv(r"C:\Users\cbush\Casino_Shit\ensembleModel\baccarat_ensemble_log.csv")

# Parse hand string into P1, P2, B1, B2
def parse_hand(entry):
    try:
        entry = str(entry)
        core = entry[:4]
        return {
            "P1": int(core[0]),
            "P2": int(core[1]),
            "B1": int(core[2]),
            "B2": int(core[3])
        }
    except:
        return {"P1": None, "P2": None, "B1": None, "B2": None}

parsed = df["hand"].apply(parse_hand).apply(pd.Series)
df = pd.concat([df, parsed], axis=1)

# Drop rows with missing values
df = df.dropna(subset=["P1", "P2", "B1", "B2"])

# Estimate hand number using index
df["hand_num"] = df.reset_index().index + 1

# Bin into early/mid/late
def shoe_phase(hand_num):
    if hand_num <= 20:
        return "early"
    elif hand_num <= 44:
        return "mid"
    else:
        return "late"

df["shoe_phase"] = df["hand_num"].apply(shoe_phase)
df["shoe_phase_encoded"] = df["shoe_phase"].map({"early": 0, "mid": 1, "late": 2})

# Label mid-shoe Banker wins as good bets
df["should_bet"] = ((df["outcome"] == 2) & (df["shoe_phase"] == "mid")).astype(int)

# Final output
df_final = df[["P1", "B1", "P2", "B2", "shoe_phase_encoded", "should_bet"]]
df_final.to_csv("baccarat_model_ready.csv", index=False)

print("✅ Cleaned and saved to baccarat_model_ready.csv")
