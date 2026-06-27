import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="AI Symptom Checker",
    page_icon="🏥",
    layout="centered"
)

# ── Dataset ───────────────────────────────────────────────────────────────────
# Each row: list of symptoms → disease
# This is a structured synthetic medical dataset based on common symptom patterns

DISEASE_DATA = {
    "Common Cold": [
        ["runny_nose","sneezing","sore_throat","cough","mild_fever","fatigue"],
        ["runny_nose","sneezing","congestion","sore_throat","watery_eyes"],
        ["sneezing","cough","sore_throat","mild_fever","headache"],
        ["runny_nose","congestion","sneezing","mild_fever","body_ache"],
        ["sore_throat","cough","runny_nose","fatigue","mild_fever"],
    ],
    "Influenza (Flu)": [
        ["high_fever","body_ache","fatigue","headache","cough","chills"],
        ["high_fever","chills","muscle_pain","fatigue","sore_throat","cough"],
        ["high_fever","headache","body_ache","fatigue","runny_nose","chills"],
        ["chills","high_fever","muscle_pain","fatigue","cough","headache"],
        ["high_fever","body_ache","chills","loss_of_appetite","fatigue","headache"],
    ],
    "COVID-19": [
        ["fever","dry_cough","fatigue","loss_of_smell","loss_of_taste","shortness_of_breath"],
        ["fever","dry_cough","loss_of_smell","loss_of_taste","headache","fatigue"],
        ["shortness_of_breath","dry_cough","fever","chest_pain","fatigue"],
        ["loss_of_smell","loss_of_taste","fever","fatigue","body_ache","dry_cough"],
        ["dry_cough","fever","shortness_of_breath","fatigue","loss_of_smell"],
    ],
    "Dengue Fever": [
        ["high_fever","severe_headache","joint_pain","muscle_pain","rash","nausea"],
        ["high_fever","rash","joint_pain","eye_pain","nausea","vomiting"],
        ["sudden_high_fever","severe_headache","rash","joint_pain","muscle_pain"],
        ["high_fever","nausea","rash","eye_pain","joint_pain","fatigue"],
        ["high_fever","rash","muscle_pain","vomiting","severe_headache","joint_pain"],
    ],
    "Malaria": [
        ["high_fever","chills","sweating","headache","nausea","vomiting"],
        ["cyclical_fever","chills","sweating","muscle_pain","fatigue","headache"],
        ["high_fever","shivering","sweating","nausea","headache","body_ache"],
        ["chills","fever","sweating","vomiting","headache","fatigue"],
        ["high_fever","chills","sweating","loss_of_appetite","nausea","muscle_pain"],
    ],
    "Typhoid": [
        ["sustained_fever","abdominal_pain","headache","loss_of_appetite","weakness"],
        ["high_fever","abdominal_pain","constipation","headache","fatigue"],
        ["sustained_fever","nausea","abdominal_pain","weakness","headache","rash"],
        ["high_fever","diarrhea","abdominal_pain","headache","loss_of_appetite"],
        ["sustained_fever","abdominal_pain","weakness","nausea","headache","fatigue"],
    ],
    "Gastroenteritis": [
        ["nausea","vomiting","diarrhea","abdominal_cramps","mild_fever","fatigue"],
        ["diarrhea","vomiting","abdominal_cramps","nausea","dehydration"],
        ["stomach_pain","diarrhea","vomiting","nausea","mild_fever"],
        ["abdominal_cramps","diarrhea","nausea","vomiting","loss_of_appetite"],
        ["diarrhea","stomach_pain","nausea","vomiting","mild_fever","fatigue"],
    ],
    "Migraine": [
        ["severe_headache","nausea","light_sensitivity","sound_sensitivity","vomiting"],
        ["throbbing_headache","nausea","light_sensitivity","blurred_vision"],
        ["severe_headache","vomiting","light_sensitivity","sound_sensitivity"],
        ["throbbing_headache","blurred_vision","nausea","light_sensitivity","fatigue"],
        ["severe_headache","visual_disturbances","nausea","light_sensitivity","sound_sensitivity"],
    ],
    "Hypertension": [
        ["headache","dizziness","blurred_vision","chest_pain","shortness_of_breath"],
        ["severe_headache","dizziness","nosebleed","shortness_of_breath","chest_pain"],
        ["headache","blurred_vision","dizziness","fatigue","chest_pain"],
        ["dizziness","headache","nosebleed","blurred_vision","shortness_of_breath"],
        ["severe_headache","fatigue","chest_pain","dizziness","blurred_vision"],
    ],
    "Diabetes (Type 2)": [
        ["frequent_urination","excessive_thirst","fatigue","blurred_vision","slow_healing"],
        ["frequent_urination","fatigue","weight_loss","blurred_vision","excessive_thirst"],
        ["excessive_thirst","frequent_urination","hunger","fatigue","slow_healing"],
        ["fatigue","blurred_vision","frequent_urination","slow_healing","numbness"],
        ["excessive_thirst","frequent_urination","fatigue","numbness","weight_loss"],
    ],
    "Pneumonia": [
        ["high_fever","cough","shortness_of_breath","chest_pain","fatigue","chills"],
        ["cough","chest_pain","shortness_of_breath","high_fever","sweating","chills"],
        ["high_fever","productive_cough","shortness_of_breath","chest_pain","fatigue"],
        ["chills","high_fever","cough","shortness_of_breath","nausea","chest_pain"],
        ["shortness_of_breath","chest_pain","high_fever","cough","fatigue","sweating"],
    ],
    "Asthma": [
        ["shortness_of_breath","wheezing","chest_tightness","cough","difficulty_breathing"],
        ["wheezing","cough","shortness_of_breath","chest_tightness"],
        ["difficulty_breathing","wheezing","chest_tightness","cough","fatigue"],
        ["shortness_of_breath","wheezing","cough","chest_tightness","anxiety"],
        ["wheezing","difficulty_breathing","cough","chest_tightness","shortness_of_breath"],
    ],
}

# ── All unique symptoms ────────────────────────────────────────────────────────
ALL_SYMPTOMS = sorted(set(
    symptom
    for symptoms_list in DISEASE_DATA.values()
    for symptoms in symptoms_list
    for symptom in symptoms
))

SYMPTOM_DISPLAY = {s: s.replace("_", " ").title() for s in ALL_SYMPTOMS}


# ── Build training data ────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    rows = []
    labels = []
    for disease, symptom_lists in DISEASE_DATA.items():
        for symptoms in symptom_lists:
            row = {s: 1 if s in symptoms else 0 for s in ALL_SYMPTOMS}
            rows.append(row)
            labels.append(disease)

    df = pd.DataFrame(rows)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, le, df.columns.tolist(), accuracy


model, label_encoder, feature_cols, accuracy = train_model()


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("AI Disease Symptom Checker")
st.write(
    "Select your symptoms below and the AI will suggest possible conditions. "
    "**This is not a medical diagnosis — always consult a doctor.**"
)

# model stats
col1, col2, col3 = st.columns(3)
col1.metric("Diseases covered", len(DISEASE_DATA))
col2.metric("Symptoms tracked", len(ALL_SYMPTOMS))
col3.metric("Model accuracy", f"{accuracy * 100:.0f}%")

st.divider()

# symptom selection by category
st.subheader("Select your symptoms")

# group symptoms into categories for easier UI
SYMPTOM_GROUPS = {
    "Fever & Temperature": [
        "fever","high_fever","mild_fever","sustained_fever","cyclical_fever","chills","sweating","shivering"
    ],
    "Head & Neurological": [
        "headache","severe_headache","throbbing_headache","dizziness","blurred_vision",
        "visual_disturbances","light_sensitivity","sound_sensitivity","numbness"
    ],
    "Respiratory": [
        "cough","dry_cough","productive_cough","shortness_of_breath","wheezing",
        "chest_tightness","difficulty_breathing","congestion","runny_nose","sneezing"
    ],
    "Throat & Nose": [
        "sore_throat","loss_of_smell","loss_of_taste","watery_eyes","nosebleed"
    ],
    "Body Pain": [
        "body_ache","muscle_pain","joint_pain","eye_pain","chest_pain","abdominal_pain",
        "abdominal_cramps","stomach_pain"
    ],
    "Digestive": [
        "nausea","vomiting","diarrhea","constipation","loss_of_appetite","dehydration"
    ],
    "Skin & General": [
        "rash","fatigue","weakness","weight_loss","slow_healing","anxiety"
    ],
    "Other": [
        "excessive_thirst","frequent_urination","hunger","sudden_high_fever"
    ],
}

selected_symptoms = []

for group, symptoms in SYMPTOM_GROUPS.items():
    available = [s for s in symptoms if s in ALL_SYMPTOMS]
    if not available:
        continue
    with st.expander(f"**{group}**", expanded=(group == "Fever & Temperature")):
        cols = st.columns(2)
        for i, symptom in enumerate(available):
            if cols[i % 2].checkbox(SYMPTOM_DISPLAY[symptom], key=symptom):
                selected_symptoms.append(symptom)

st.divider()

# predict button
if st.button("Check symptoms", type="primary", use_container_width=True):
    if len(selected_symptoms) < 2:
        st.warning("Please select at least 2 symptoms for a meaningful result.")
    else:
        input_vector = {s: 1 if s in selected_symptoms else 0 for s in feature_cols}
        input_df = pd.DataFrame([input_vector])

        probabilities = model.predict_proba(input_df)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]

        st.subheader("Possible conditions")

        for rank, idx in enumerate(top_indices):
            prob = probabilities[idx]
            if prob < 0.05:
                continue
            disease = label_encoder.inverse_transform([idx])[0]

            if rank == 0:
                color = "🔴"
                label = "Most likely"
            elif rank == 1:
                color = "🟡"
                label = "Possible"
            else:
                color = "🟢"
                label = "Less likely"

            with st.container():
                st.markdown(f"**{color} {label}: {disease}**")
                st.progress(float(prob))
                st.caption(f"Confidence: {prob * 100:.1f}%")

        # show selected symptoms
        st.divider()
        st.subheader("Symptoms you selected")
        symptom_tags = " · ".join([SYMPTOM_DISPLAY[s] for s in selected_symptoms])
        st.write(symptom_tags)

        # important disclaimer
        st.error(
            "⚠️ This AI tool is for educational purposes only. "
            "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
            "Always consult a qualified doctor for any health concerns."
        )

st.divider()
st.caption(
    "Built with Streamlit and Scikit-learn Random Forest | "
    f"Covers {len(DISEASE_DATA)} diseases and {len(ALL_SYMPTOMS)} symptoms"
)