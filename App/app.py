"""Streamlit research demo for the versioned production pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from asd_screening.config import AQ_FEATURES, MODEL_PATH  # noqa: E402
from asd_screening.inference import ArtifactError, load_artifact, predict_record  # noqa: E402

st.set_page_config(
    page_title="Adult ASD Screening — ML Research Demo",
    page_icon="ASD",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --accent: #22d3ee; }
      .stApp { background: #07090b; color: #f4f7f8; }
      [data-testid="stSidebar"] { background: #0b0f12; border-right: 1px solid #20272d; }
      [data-testid="stForm"] { border: 1px solid #242c32; border-radius: 12px; padding: 1.2rem; }
      .block-container { max-width: 1120px; padding-top: 2.5rem; }
      .eyebrow {
        color: #67e8f9; font-size: .78rem; font-weight: 700;
        letter-spacing: .14em; text-transform: uppercase;
      }
      .disclaimer {
        border-left: 3px solid #f59e0b; background: #17130a;
        padding: .85rem 1rem; border-radius: 4px;
      }
      .result-positive { border-left: 3px solid #f59e0b; background: #17130a; padding: 1rem; }
      .result-negative { border-left: 3px solid #22d3ee; background: #071518; padding: 1rem; }
      div.stButton > button { border-radius: 8px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cached_artifact(path: str) -> dict[str, Any]:
    return load_artifact(path)


def display_value(value: str) -> str:
    labels = {
        "f": "Female",
        "m": "Male",
        "yes": "Yes",
        "no": "No",
        "Self": "Self",
    }
    return labels.get(value, value)


try:
    artifact = cached_artifact(str(MODEL_PATH))
except ArtifactError as error:
    st.error(str(error))
    st.code("python -m asd_screening.training", language="bash")
    st.stop()

metadata = artifact["metadata"]
category_options: dict[str, list[str]] = metadata["category_options"]

with st.sidebar:
    st.markdown('<p class="eyebrow">Model card</p>', unsafe_allow_html=True)
    st.subheader(metadata["model_label"])
    st.caption(f"Artifact version {metadata['project_version']}")
    st.markdown(
        "This demo runs a complete scikit-learn pipeline: input validation, train-fitted "
        "imputation, one-hot encoding, scaling, and classical classification."
    )
    with st.expander("Technical contract"):
        st.write(f"**Scope:** {metadata['scope']}")
        st.write(f"**Decision threshold:** {artifact['threshold']:.3f}")
        st.write(f"**Features:** {len(metadata['features'])}")
        st.write(f"**Generated:** {metadata['generated_at_utc'][:10]}")
    st.markdown(
        '<div class="disclaimer"><strong>Educational use only.</strong><br>'
        "This model is not a medical device, diagnosis, or substitute for evaluation by a "
        "qualified clinician.</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<p class="eyebrow">Classical machine learning · research demo</p>',
    unsafe_allow_html=True,
)
st.title("Adult ASD screening classification")
st.markdown(
    "An auditable inference interface for the repository's production pipeline. "
    "It accepts the same feature contract used during evaluation and clearly separates "
    "a model output from a clinical conclusion."
)

with st.expander("Why the screening items use 0/1 scores", expanded=False):
    st.write(
        "The supplied dataset contains ten already-scored binary screening fields but does not "
        "bundle authoritative questionnaire wording or scoring documentation. To avoid inventing "
        "a clinical instrument, this interface asks for the recorded 0/1 feature values directly."
    )

with st.form("screening_form", clear_on_submit=False, border=True):
    st.subheader("1. Scored screening features")
    st.caption("Choose the binary value recorded for each source feature.")
    item_columns = st.columns(5)
    answers: dict[str, int] = {}
    for index, feature in enumerate(AQ_FEATURES):
        with item_columns[index % 5]:
            answers[feature] = st.selectbox(
                feature.replace("_Score", ""),
                options=[0, 1],
                format_func=lambda value: f"{value} — {'positive' if value else 'negative'} score",
                key=f"score_{feature}",
            )

    st.subheader("2. Demographic and background features")
    left, middle, right = st.columns(3)
    with left:
        age = st.number_input("Age", min_value=18, max_value=100, value=25, step=1)
        gender = st.selectbox(
            "Gender as represented in the dataset",
            category_options["gender"],
            format_func=display_value,
        )
        ethnicity = st.selectbox(
            "Ethnicity", category_options["ethnicity"], format_func=display_value
        )
    with middle:
        jaundice = st.selectbox(
            "Jaundice at birth",
            category_options["jaundice"],
            format_func=display_value,
        )
        family_history = st.selectbox(
            "Family history of ASD",
            category_options["family_asd_history"],
            format_func=display_value,
        )
        used_before = st.selectbox(
            "Previously used a screening app",
            category_options["used_app_before"],
            format_func=display_value,
        )
    with right:
        country = st.selectbox("Country of residence", category_options["country_of_residence"])
        relation = st.selectbox(
            "Person completing the screening",
            category_options["relation"],
            format_func=display_value,
        )

    submitted = st.form_submit_button(
        "Run research inference", type="primary", use_container_width=True
    )

if submitted:
    values: dict[str, Any] = {
        **answers,
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "jaundice": jaundice,
        "family_asd_history": family_history,
        "country_of_residence": country,
        "used_app_before": used_before,
        "relation": relation,
    }
    try:
        outcome = predict_record(artifact, values)
    except (TypeError, ValueError) as error:
        st.error(f"The record could not be evaluated: {error}")
    else:
        st.subheader("Research model output")
        if outcome["class"] == 1:
            st.markdown(
                '<div class="result-positive"><strong>Positive model classification</strong><br>'
                "The feature pattern crossed the threshold learned from training-only predictions. "
                "This is not evidence that a person has ASD.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-negative"><strong>Negative model classification</strong><br>'
                "The feature pattern did not cross the learned threshold. This cannot rule out ASD "
                "and is not a clinical assessment.</div>",
                unsafe_allow_html=True,
            )
        with st.expander("Technical output"):
            st.metric("Model score", f"{outcome['score']:.3f}")
            st.caption(
                f"Decision threshold: {outcome['threshold']:.3f}. The score is an uncalibrated "
                "model "
                "output and must not be interpreted as an individual clinical probability."
            )

st.divider()
st.caption(
    "Repository research demo · No responses are stored or transmitted by this application · "
    "See MODEL_CARD.md and DATA_CARD.md for evaluation and limitations."
)
