"""
Stress Level Predictor — Sleep Health & Lifestyle
-------------------------------------------------
Streamlit app built on `model_prod.pkl`, a joblib dict with:
    Model  : KNeighborsClassifier
    OHE    : OneHotEncoder  (Gender, Occupation, BMI Category, Sleep Disorder, Blood Pressure)
    Scaler : MinMaxScaler   (Age, Physical Activity Level, Heart Rate, Daily Steps)
    LE     : LabelEncoder   (target: Stress Level 3..8 -> 0..5)

Feature pipeline (identical to training):
    1. Sleep Disorder: missing -> "Others"
    2. One-hot encode the 5 categorical columns
    3. Min-max scale the 4 numeric columns
    4. Append Sleep Duration and Quality of Sleep unscaled
    5. KNN predict -> decode with LabelEncoder

Run:  streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_prod.pkl"
DATA_PATH = BASE_DIR / "Sleep_health_and_lifestyle_dataset.csv"

RAW_NUMERIC = ["Sleep Duration", "Quality of Sleep"]  # not scaled during training
RAW_RANGES = {"Sleep Duration": (5.8, 8.5), "Quality of Sleep": (4, 9)}  # seen in training data

st.set_page_config(
    page_title="Stress Level Predictor",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}

.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
    border-radius: 20px;
    padding: 2rem 2.2rem;
    color: #fff;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(79, 70, 229, .25);
}
.hero h1 {margin: 0 0 .3rem 0; font-size: 2.1rem; color: #fff; padding: 0;}
.hero p  {margin: 0; font-size: 1.02rem; opacity: .92;}

.result-card {
    border-radius: 18px;
    padding: 1.6rem 1.4rem;
    text-align: center;
    border: 1px solid rgba(128,128,128,.25);
}
.result-card .label {font-size: .85rem; letter-spacing: .08em; text-transform: uppercase; opacity: .75;}
.result-card .score {font-size: 4rem; font-weight: 800; line-height: 1.1; margin: .2rem 0;}
.result-card .band  {font-size: 1.2rem; font-weight: 700;}
.result-card .msg   {font-size: .95rem; opacity: .85; margin-top: .6rem;}
.low  {background: rgba(34,197,94,.12);  color: #16a34a;}
.mid  {background: rgba(245,158,11,.13); color: #d97706;}
.high {background: rgba(239,68,68,.12);  color: #dc2626;}

.stat {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    padding: .9rem 1rem;
    text-align: center;
}
.stat .v {font-size: 1.5rem; font-weight: 700;}
.stat .k {font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: .06em;}

div[data-testid="stForm"] {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 18px;
    padding: 1.3rem 1.4rem;
}
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: #fff; border: 0; border-radius: 12px;
    padding: .65rem 1.4rem; font-weight: 600; width: 100%;
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    filter: brightness(1.08); color: #fff;
}
.footer {text-align: center; opacity: .6; font-size: .8rem; margin-top: 2.5rem;}
#MainMenu, footer {visibility: hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts(path: Path) -> dict:
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_dataset(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Feature engineering + prediction
# --------------------------------------------------------------------------- #
def parse_bp(label: str) -> tuple[int, int]:
    s, d = label.split("/")
    return int(s), int(d)


def nearest_bp_category(systolic: int, diastolic: int, categories) -> str:
    """The OHE only knows 24 exact BP readings seen in training (unknown -> all zeros).
    Snap the entered reading to the closest known one so the input is never silently dropped."""
    return min(
        categories,
        key=lambda c: (parse_bp(c)[0] - systolic) ** 2 + (parse_bp(c)[1] - diastolic) ** 2,
    )


def build_features(art: dict, row: dict) -> pd.DataFrame:
    ohe, scaler, model = art["OHE"], art["Scaler"], art["Model"]
    cat_cols = list(ohe.feature_names_in_)
    scaled_cols = list(scaler.feature_names_in_)

    df = pd.DataFrame([row])
    encoded = np.asarray(ohe.transform(df[cat_cols]))
    scaled = scaler.transform(df[scaled_cols])
    raw = df[RAW_NUMERIC].to_numpy(dtype=float)

    X = np.hstack([encoded, scaled, raw])
    return pd.DataFrame(X, columns=list(model.feature_names_in_))


def predict(art: dict, row: dict):
    model, le = art["Model"], art["LE"]
    X = build_features(art, row)
    pred_encoded = model.predict(X)
    stress = int(le.inverse_transform(pred_encoded)[0])
    proba = pd.Series(
        model.predict_proba(X)[0],
        index=[int(v) for v in le.inverse_transform(model.classes_)],
        name="Share of neighbours",
    )
    return stress, proba


def stress_band(level: int):
    if level <= 4:
        return "low", "Low stress", "Your profile looks similar to people who report low stress. Keep up the routine."
    if level <= 6:
        return "mid", "Moderate stress", "Your profile sits in the middle range. Small changes in sleep or activity may help."
    return "high", "High stress", "Your profile resembles people reporting high stress. Consider sleep, rest and activity habits."


def range_warnings(art: dict, row: dict) -> list[str]:
    scaler = art["Scaler"]
    limits = {c: (lo, hi) for c, lo, hi in zip(scaler.feature_names_in_, scaler.data_min_, scaler.data_max_)}
    limits.update(RAW_RANGES)
    out = []
    for col, (lo, hi) in limits.items():
        if not (lo <= row[col] <= hi):
            out.append(f"**{col}** = {row[col]} is outside the training range ({lo:g}–{hi:g}).")
    return out


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
        <h1>😴 Stress Level Predictor</h1>
        <p>Enter your sleep, health and lifestyle details and a K-Nearest-Neighbours model
        estimates your stress level, based on the Sleep Health &amp; Lifestyle dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(f"`{MODEL_PATH.name}` was not found. Put it in the same folder as `app.py`.")
    st.stop()

art = load_artifacts(MODEL_PATH)
df_data = load_dataset(DATA_PATH)

ohe_cats = dict(zip(art["OHE"].feature_names_in_, art["OHE"].categories_))
SLEEP_DISORDER_LABELS = {"Others": "None"}  # "None" is stored as "Others" in the model

tab_predict, tab_insights, tab_about = st.tabs(["🔮 Predict", "📊 Dataset insights", "ℹ️ About the model"])

# --------------------------------------------------------------------------- #
# Predict tab
# --------------------------------------------------------------------------- #
with tab_predict:
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        with st.form("input_form"):
            st.subheader("Your details")

            c1, c2 = st.columns(2)
            gender = c1.selectbox("Gender", ohe_cats["Gender"])
            age = c2.number_input("Age", min_value=18, max_value=80, value=35, step=1)

            occupation = st.selectbox("Occupation", ohe_cats["Occupation"])

            c3, c4 = st.columns(2)
            bmi = c3.selectbox(
                "BMI category",
                ohe_cats["BMI Category"],
                index=list(ohe_cats["BMI Category"]).index("Normal Weight")
                if "Normal Weight" in ohe_cats["BMI Category"]
                else 0,
            )
            disorder_opts = list(ohe_cats["Sleep Disorder"])
            disorder_disp = [SLEEP_DISORDER_LABELS.get(o, o) for o in disorder_opts]
            disorder_label = c4.selectbox("Sleep disorder", disorder_disp, index=disorder_opts.index("Others"))
            sleep_disorder = disorder_opts[disorder_disp.index(disorder_label)]

            st.markdown("**Blood pressure (mmHg)**")
            c5, c6 = st.columns(2)
            systolic = c5.number_input("Systolic", min_value=80, max_value=200, value=120, step=1)
            diastolic = c6.number_input("Diastolic", min_value=50, max_value=130, value=80, step=1)

            st.markdown("**Sleep**")
            c7, c8 = st.columns(2)
            sleep_duration = c7.slider("Sleep duration (hours)", 3.0, 12.0, 7.0, 0.1)
            sleep_quality = c8.slider("Quality of sleep (1–10)", 1, 10, 7)

            st.markdown("**Activity & vitals**")
            c9, c10, c11 = st.columns(3)
            activity = c9.number_input("Physical activity (min/day)", 0, 180, 45, 5)
            heart_rate = c10.number_input("Heart rate (bpm)", 40, 140, 70, 1)
            steps = c11.number_input("Daily steps", 0, 25000, 7000, 100)

            submitted = st.form_submit_button("Predict stress level")

    with right:
        if submitted:
            bp_label = nearest_bp_category(int(systolic), int(diastolic), ohe_cats["Blood Pressure"])
            row = {
                "Gender": gender,
                "Age": age,
                "Occupation": occupation,
                "Sleep Duration": float(sleep_duration),
                "Quality of Sleep": sleep_quality,
                "Physical Activity Level": activity,
                "BMI Category": bmi,
                "Blood Pressure": bp_label,
                "Heart Rate": heart_rate,
                "Daily Steps": steps,
                "Sleep Disorder": sleep_disorder,
            }

            try:
                level, proba = predict(art, row)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")
            else:
                css, band, msg = stress_band(level)
                st.markdown(
                    f"""
                    <div class="result-card {css}">
                        <div class="label">Predicted stress level</div>
                        <div class="score">{level}</div>
                        <div class="band">{band}</div>
                        <div class="msg">{msg}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                k = art["Model"].n_neighbors
                st.markdown(f"##### How the {k} nearest neighbours voted")
                st.bar_chart(proba, height=220)
                st.caption(
                    f"Blood pressure {int(systolic)}/{int(diastolic)} was matched to the closest "
                    f"reading in the training data ({bp_label})."
                )

                for w in range_warnings(art, row):
                    st.warning(w + " Treat this prediction with extra caution.")
        else:
            st.markdown(
                """
                <div class="result-card" style="min-height: 260px; display:flex; flex-direction:column; justify-content:center;">
                    <div class="label">Result</div>
                    <div class="score">–</div>
                    <div class="msg">Fill in the form and press <b>Predict stress level</b>.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "This is a demo built on a small, partly synthetic dataset. "
            "It is not medical advice or a diagnosis."
        )

# --------------------------------------------------------------------------- #
# Insights tab
# --------------------------------------------------------------------------- #
with tab_insights:
    if df_data is None:
        st.info(f"Put `{DATA_PATH.name}` next to `app.py` to see dataset insights.")
    else:
        d = df_data.copy()
        s1, s2, s3, s4 = st.columns(4)
        for col, val, key in [
            (s1, f"{len(d):,}", "Records"),
            (s2, f"{d['Stress Level'].mean():.1f}", "Avg stress level"),
            (s3, f"{d['Sleep Duration'].mean():.1f} h", "Avg sleep"),
            (s4, f"{d['Quality of Sleep'].mean():.1f}/10", "Avg sleep quality"),
        ]:
            col.markdown(f'<div class="stat"><div class="v">{val}</div><div class="k">{key}</div></div>', unsafe_allow_html=True)

        st.write("")
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("##### Stress level distribution")
            st.bar_chart(d["Stress Level"].value_counts().sort_index(), height=280)
        with b:
            st.markdown("##### Average stress by occupation")
            st.bar_chart(d.groupby("Occupation")["Stress Level"].mean().sort_values(), height=280)

        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("##### Average stress by sleep quality")
            st.line_chart(d.groupby("Quality of Sleep")["Stress Level"].mean(), height=280)
        with b:
            st.markdown("##### Average stress by sleep duration")
            st.line_chart(d.assign(h=d["Sleep Duration"].round(0)).groupby("h")["Stress Level"].mean(), height=280)

        with st.expander("Preview dataset"):
            st.dataframe(d, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# About tab
# --------------------------------------------------------------------------- #
with tab_about:
    model = art["Model"]
    st.markdown(
        f"""
        ### Model
        - **Algorithm:** K-Nearest Neighbours classifier (`k = {model.n_neighbors}`, `{model.weights}` weights, `{model.metric}` metric)
        - **Target:** Stress Level (classes {", ".join(str(int(v)) for v in art["LE"].classes_)}), encoded with `LabelEncoder`
        - **Input features after engineering:** {model.n_features_in_}

        ### Feature engineering
        1. Missing *Sleep Disorder* values are treated as the category `Others` ("None" in this app).
        2. *Gender, Occupation, BMI Category, Sleep Disorder, Blood Pressure* are one-hot encoded.
        3. *Age, Physical Activity Level, Heart Rate, Daily Steps* are scaled with `MinMaxScaler`.
        4. *Sleep Duration* and *Quality of Sleep* are passed through as-is.

        ### Notes
        - Blood pressure is a categorical feature in this model, so the reading you enter is matched to the nearest one seen in training.
        - Predictions are most reliable inside the training ranges (the app warns you otherwise).
        - Stress bands (low 3–4, moderate 5–6, high 7–8) are a simple grouping added for readability.
        """
    )

st.markdown('<div class="footer">Built with Streamlit · scikit-learn</div>', unsafe_allow_html=True)
