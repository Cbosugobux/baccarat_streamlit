# baccarat_next_bet_app.py

import streamlit as st
import pandas as pd
import pickle
import os

MODEL_PATH = "../baccarat_rf_model.pkl"
SIDE_MODEL_PATH = "../baccarat_side_model.pkl"
LOG_PATH = "LastChance/baccarat_live_log.csv"

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_side_model():
    with open(SIDE_MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()
side_model = load_side_model()

# Helper function to classify shoe phase
def classify_shoe_phase(hand_num):
    if hand_num <= 15:
        return 0
    elif hand_num <= 50:
        return 1
    else:
        return 2

st.title("🎰 Baccarat Next Bet Predictor")

if "hand" not in st.session_state:
    st.session_state.hand = None
    st.session_state.records = []

col1, col2 = st.columns([2, 1])

with col1:
    # New shoe setup
    if st.button("🔄 Start New Shoe"):
        st.session_state.hand = None
        st.session_state.records = []
        st.success("New shoe started. Enter the first hand.")

    if st.session_state.hand is None:
        hand_input = st.number_input("Enter starting hand number:", min_value=1, step=1, key="start_hand")
        if st.button("Set Starting Hand"):
            st.session_state.hand = hand_input
    else:
        st.subheader(f"Hand #{st.session_state.hand}")

        fast_input = st.text_input("Enter hand as P1P2B1B2B3P3O (e.g., 2035012)", key="fast_input")
        if st.button("Submit Hand") and len(fast_input) == 7 and fast_input.isdigit():
            p1 = int(fast_input[0])
            p2 = int(fast_input[1])
            b1 = int(fast_input[2])
            b2 = int(fast_input[3])
            b3 = int(fast_input[4])
            p3 = int(fast_input[5])
            outcome_digit = int(fast_input[6])
            outcome = {0: "P", 1: "T", 2: "B"}.get(outcome_digit, "")

            shoe_phase = classify_shoe_phase(st.session_state.hand)
            model_input = pd.DataFrame([[p1, b1, p2, b2, shoe_phase]], columns=["P1", "B1", "P2", "B2", "shoe_phase_encoded"])

            prediction = model.predict(model_input)[0]
            pred_proba = model.predict_proba(model_input)[0][1]
            pred_text = "✅ BET on next hand" if prediction == 1 else "🚫 Do NOT bet next hand"
            st.markdown(f"### Prediction: {pred_text} (Confidence: {pred_proba:.2f})")

            if prediction == 1:
                side_pred = side_model.predict(model_input)[0]
                side_proba = side_model.predict_proba(model_input)[0]
                banker_proba = side_proba[1]
                player_proba = side_proba[0]
                side_text = "🔵 Bet Banker" if side_pred == 1 else "🔴 Bet Player"
                st.markdown(f"### Suggested Side: {side_text} — Banker: {banker_proba:.2f}, Player: {player_proba:.2f}")
            else:
                side_text = "---"

            st.session_state.records.append({
                "hand": st.session_state.hand,
                "P1": p1,
                "P2": p2,
                "B1": b1,
                "B2": b2,
                "P3": p3,
                "B3": b3,
                "player_3rd": p3,
                "banker_3rd": b3,
                "xgb_prob": f"{pred_proba:.2f}",
                "nn_prob": "",
                "bet_made": prediction,
                "unit_size": "",
                "result": "",
                "outcome": outcome,
                "bankroll": ""
            })
            st.session_state.hand += 1

    # Option to export
    if st.session_state.records:
        if st.button("💾 Save Session Log"):
            log_df = pd.DataFrame(st.session_state.records)
            if os.path.exists(LOG_PATH):
                log_df.to_csv(LOG_PATH, mode='a', header=False, index=False)
            else:
                log_df.to_csv(LOG_PATH, index=False)
            st.success(f"Saved {len(log_df)} hands to {LOG_PATH}")

with col2:
    if st.session_state.records:
        st.subheader("📋 Past Hands")
        history_df = pd.DataFrame(st.session_state.records)
        st.dataframe(history_df.tail(10), use_container_width=True)
