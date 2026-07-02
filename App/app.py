import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path

st.set_page_config(page_title="ASD Screening Tool", layout="wide")

try:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parent.parent
    MODEL_PATH = project_dir / 'Models' / 'best_model.joblib'
except NameError:
    MODEL_PATH = Path('Models') / 'best_model.joblib'

# This function loads the pre-trained model and caches it so it doesn't reload on every interaction.
@st.cache_resource
def load_model(path):
    """Loads the joblib model file safely."""
    try:
        if os.path.exists(path):
            model_info = joblib.load(path)
            if 'model' in model_info:
                return model_info['model']
            else:
                st.error("Error: 'model' key not found in the loaded joblib file.")
                return None
        else:
            st.error(f"Error: Model file not found at the specified path: {path}")
            return None
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        return None

# Load the model into the app
model = load_model(MODEL_PATH)

# These dictionaries map the user-friendly input options to the numerical values the model expects.
ethnicity_map = {
    "Asian": 0.0597, "Black": 0.1277, "Hispanic": 0.2222, "Latino": 0.2353,
    "Middle Eastern": 0.0619, "Others": 0.0383, "Pasifika": 0.1875,
    "South Asian": 0.0882, "Turkish": 0.0000, "White-European": 0.4708,
}
relation_map = {
    "Self": 0.2073, "Parent": 0.1538, "Relative": 0.1538,
    "Health care professional": 0.1538, "Others": 0.1538,
}
region_map = {
    'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
    'Nicaragua': 'North America', 'AmericanSamoa': 'North America', 'Bahamas': 'North America',
    'Brazil': 'South America', 'Argentina': 'South America', 'Bolivia': 'South America',
    'United Kingdom': 'Europe', 'Austria': 'Europe', 'Ukraine': 'Europe',
    'France': 'Europe', 'Netherlands': 'Europe', 'Italy': 'Europe',
    'Ireland': 'Europe', 'Russia': 'Europe', 'Serbia': 'Europe',
    'Sweden': 'Europe', 'Iceland': 'Europe', 'Germany': 'Europe',
    'Spain': 'Europe', 'Czech Republic': 'Europe', 'Romania': 'Europe',
    'Cyprus': 'Europe', 'Belgium': 'Europe',
    'Jordan': 'Middle East', 'United Arab Emirates': 'Middle East', 'Iraq': 'Middle East',
    'Oman': 'Middle East', 'Saudi Arabia': 'Middle East', 'Iran': 'Middle East',
    'Azerbaijan': 'Middle East', 'Armenia': 'Middle East',
    'India': 'Asia', 'Malaysia': 'Asia', 'Vietnam': 'Asia',
    'Sri Lanka': 'Asia', 'Hong Kong': 'Asia', 'China': 'Asia',
    'Pakistan': 'Asia', 'Japan': 'Asia', 'Bangladesh': 'Asia',
    'South Africa': 'Africa', 'Egypt': 'Africa', 'Ethiopia': 'Africa',
    'Angola': 'Africa', 'Sierra Leone': 'Africa', 'Niger': 'Africa',
    'Burundi': 'Africa',
    'New Zealand': 'Oceania', 'Australia': 'Oceania', 'Tonga': 'Oceania',
    'Kazakhstan': 'Central Asia',
    'Aruba': 'Caribbean'
}
region_frequencies = {
    'Africa': 0.1875, 'Asia': 0.0612, 'Caribbean': 0.5000, 'Central Asia': 0.3000,
    'Europe': 0.2470, 'Middle East': 0.0909, 'North America': 0.4341,
    'Oceania': 0.0690, 'South America': 0.1250, 'Other': 0.1364
}

# This holds the text and scoring logic for the 10 screening questions.
aq_questions_config = {
    "A1": {"text": "I often notice small sounds when others do not.", "scoring": "forward"},
    "A2": {"text": "When I’m reading a story, I find it difficult to work out the characters’ intentions.", "scoring": "forward"},
    "A3": {"text": "I find it easy to 'read between the lines' when someone is talking to me.", "scoring": "reverse"},
    "A4": {"text": "I usually concentrate more on the whole picture, rather than the small details.", "scoring": "reverse"},
    "A5": {"text": "I know how to tell if someone listening to me is getting bored.", "scoring": "reverse"},
    "A6": {"text": "I find it easy to do more than one thing at once.", "scoring": "reverse"},
    "A7": {"text": "I find it easy to work out what someone is thinking or feeling just by looking at their face.", "scoring": "reverse"},
    "A8": {"text": "If there is an interruption, I can switch back to what I was doing very quickly.", "scoring": "reverse"},
    "A9": {"text": "I like to collect information about categories of things.", "scoring": "forward"},
    "A10": {"text": "I find it difficult to work out people’s intentions.", "scoring": "forward"},
}
aq_options = ["Definitely Disagree", "Slightly Disagree", "Slightly Agree", "Definitely Agree"]

# This list ensures that the features are passed to the model in the correct order.
EXPECTED_FEATURE_NAMES = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'jaundice', 'austim', 'used_app_before', 'result',
    'ethnicity_target', 'relation_target', 'region_target'
]

# A default text for dropdowns to make sure the user selects an option.
placeholder_text = "--- Select an Option ---"

# The sidebar contains general information and the disclaimer.
with st.sidebar:
    st.title("About")
    st.info(
        "This application uses a machine learning model to screen for Autism Spectrum Disorder (ASD) "
        "based on the AQ-10 questionnaire and demographic data. "
        "The model is a Support Vector Classifier (SVC) selected through hyperparameter tuning."
    )
    st.title("Disclaimer")
    st.warning(
        "This is a screening tool and **not** a diagnostic assessment. "
        "The results are not a substitute for a professional medical opinion. "
        "Please consult a qualified healthcare provider for an accurate diagnosis and for any health concerns."
    )

# This is the main part of the application.
st.title("Autism Spectrum Disorder (ASD) Screening Tool")
st.markdown("Please answer the following questions based on how you typically feel or behave. This is a screening tool and **not** a diagnostic assessment.")
st.markdown("---")

# If the model didn't load, stop the app from running further.
if model is None:
    st.error("Critical Error: The prediction model could not be loaded. Please ensure the 'Models/best_model.joblib' file exists and is valid. The application cannot proceed.")
    st.stop()

# This dictionary will store all the answers from the user.
user_inputs = {}

# The screening questions are presented in a bordered container.
with st.container(border=True):
    st.subheader("📝 Screening Questions")
    st.caption("Please select the option that best describes you for each statement.")
    for i, (key_suffix, config) in enumerate(aq_questions_config.items()):
        question_key = f"aq_{key_suffix}"
        input_key = key_suffix + "_Score"
        user_inputs[input_key] = st.radio(
            f"**{i+1}. {config['text']}**",
            options=aq_options,
            key=question_key,
            horizontal=True,
            index=None
        )

# The demographic questions are in a separate container, organized into two columns.
with st.container(border=True):
    st.subheader("👤 Demographic & Background Information")
    
    col1, col2 = st.columns(2)
    with col1:
        user_inputs['age'] = st.number_input("Age (years):", min_value=1, max_value=120, value=None, step=1, placeholder="Enter your age")
        user_inputs['jaundice'] = st.radio("Born with Jaundice?", options=["No", "Yes"], key="jaundice_radio", index=None, horizontal=True)
        user_inputs['used_app_before'] = st.radio("Used a screening app like this before?", options=["No", "Yes"], key="used_app_radio", index=None, horizontal=True)
        user_inputs['ethnicity'] = st.selectbox("Ethnicity:", options=[placeholder_text] + list(ethnicity_map.keys()), index=0)

    with col2:
        user_inputs['gender'] = st.radio("Gender:", options=["Female", "Male"], key="gender_radio", index=None, horizontal=True)
        user_inputs['austim'] = st.radio("Family member with ASD?", options=["No", "Yes"], key="autism_radio", index=None, horizontal=True)
        user_inputs['relation'] = st.selectbox("Relation to person being screened:", options=[placeholder_text] + list(relation_map.keys()), index=0)
        
        country_options = [placeholder_text] + sorted(list(region_map.keys()))
        selected_country = st.selectbox("Country of Residence:", options=country_options, index=0)

st.markdown("---")
submit_button = st.button("Get Screening Result", type="primary", use_container_width=True)

if submit_button:
    # First, check if all inputs have been provided.
    missing_fields = []
    for i in range(1, 11):
        if user_inputs[f"A{i}_Score"] is None:
            missing_fields.append(f"Question {i}")
            
    if user_inputs['age'] is None: missing_fields.append("Age")
    if user_inputs['gender'] is None: missing_fields.append("Gender")
    if user_inputs['jaundice'] is None: missing_fields.append("Jaundice")
    if user_inputs['austim'] is None: missing_fields.append("Family member with ASD")
    if user_inputs['used_app_before'] is None: missing_fields.append("Used app before")
    if user_inputs['ethnicity'] == placeholder_text: missing_fields.append("Ethnicity")
    if user_inputs['relation'] == placeholder_text: missing_fields.append("Relation")
    if selected_country == placeholder_text: missing_fields.append("Country of Residence")

    if missing_fields:
        st.error(f"Please complete all fields before submitting. Missing: {', '.join(missing_fields)}")
    else:
        # If validation passes, process the data and run the prediction.
        try:
            # Score the AQ-10 questions.
            a_scores = []
            agree_responses = ["Slightly Agree", "Definitely Agree"]
            disagree_responses = ["Slightly Disagree", "Definitely Disagree"]

            for i in range(1, 11):
                config = aq_questions_config[f"A{i}"]
                value = user_inputs[f"A{i}_Score"]
                score = 0
                if (config['scoring'] == 'forward' and value in agree_responses) or \
                   (config['scoring'] == 'reverse' and value in disagree_responses):
                    score = 1
                a_scores.append(score)
            
            result_sum = sum(a_scores)

            # Convert all user inputs into the numeric format required by the model.
            region = region_map[selected_country]
            
            features_in_order = [
                *a_scores,
                float(user_inputs['age']),
                1 if user_inputs['gender'] == 'Male' else 0,
                1 if user_inputs['jaundice'] == 'Yes' else 0,
                1 if user_inputs['austim'] == 'Yes' else 0,
                1 if user_inputs['used_app_before'] == 'Yes' else 0,
                result_sum,
                ethnicity_map[user_inputs['ethnicity']],
                relation_map[user_inputs['relation']],
                region_frequencies[region]
            ]

            # Create a DataFrame, as the model expects it in this format.
            input_df = pd.DataFrame([features_in_order], columns=EXPECTED_FEATURE_NAMES)

            # Get the prediction and confidence score from the model.
            prediction = model.predict(input_df)
            predicted_class = prediction[0]
            
            confidence_score = "Not available"
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_df)
                prob_confidence = probabilities[0][predicted_class]
                confidence_score = f"{prob_confidence:.1%}"

            # Display the final results.
            st.subheader("📈 Screening Outcome")
            result_col, score_col = st.columns([2, 1])

            with result_col:
                if predicted_class == 1:
                    st.warning("#### Result: Potential indicators of ASD detected.")
                    st.markdown("Based on the screening answers, there are patterns that may suggest the presence of Autism Spectrum Disorder traits. A follow-up with a healthcare professional is recommended.")
                else:
                    st.success("#### Result: Does not show significant indicators of ASD.")
                    st.markdown("Based on the screening answers, significant indicators for Autism Spectrum Disorder were not detected.")

            with score_col:
                st.metric(label="AQ-10 Score", value=f"{result_sum} / 10")
                st.info(f"**Model Confidence:** {confidence_score}")

            st.caption("Remember, this is not a diagnosis. Please see the disclaimer in the sidebar.")

        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")
            print(f"Prediction Error details: {e}")