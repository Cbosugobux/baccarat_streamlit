# clean_baccarat_log.py

import pandas as pd

# Load raw ensemble log
df = pd.read_csv(r"C:\Users\cbush\Casino_Shit\ensembleModel\baccarat_ensemble_log.csv")

# Parse the 4-digit 'cards' field: P1, P2, B1, B2
def parse_cards(card_str):
    try:
        s = str(card_str).strip()
        if len(s) == 4 and all(c.isdigit() for c in s):
            # You originally entered cards as P1P2B1B2 but we need P1B1P2B2
            return pd.Series([int(s[0]), int(s[2]), int(s[1]), int(s[3])])
        else:
            return pd.Series([None, None, None, None])
    except Exception:
        return pd.Series([None, None, None, None])


# Apply parsing to extract cards correctly
df[['P1', 'B1', 'P2', 'B2']] = df['cards'].apply(parse_cards)


# Clean and fix third cards (already provided separately)
df['P3'] = df['player_3rd'].fillna(0).astype(int)
df['B3'] = df['banker_3rd'].fillna(0).astype(int)

# Convert outcome column to integer
df['outcome'] = df['outcome'].fillna(-1).astype(int)

# Select and reorder relevant columns
cleaned_df = df[['hand', 'P1', 'B1', 'P2', 'B2', 'P3', 'B3', 'player_3rd', 'banker_3rd',
                 'xgb_prob', 'nn_prob', 'bet_made', 'unit_size', 'result', 'outcome', 'bankroll']]

# Drop rows with any missing card data
cleaned_df = cleaned_df.dropna(subset=['P1', 'B1', 'P2', 'B2'])

# Save cleaned version
cleaned_df.to_csv("baccarat_log_cleaned.csv", index=False)

print("✅ Saved cleaned file as baccarat_log_cleaned.csv")
