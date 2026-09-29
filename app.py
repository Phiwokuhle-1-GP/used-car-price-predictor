import streamlit as st
import pandas as pd
import pickle


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main page */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Blue project header */
.project-header {
    background: linear-gradient(135deg, #0B3D70, #1769AA);
    padding: 35px;
    border-radius: 14px;
    color: white;
    margin-bottom: 25px;
}

.project-header h1 {
    color: white;
    margin: 0;
    font-size: 38px;
}

.project-header p {
    color: #EAF3FA;
    font-size: 17px;
    margin-top: 10px;
}

/* Prediction card */
.price-card {
    background: #EAF4FF;
    border-left: 7px solid #1769AA;
    padding: 28px;
    border-radius: 12px;
    margin-top: 20px;
    margin-bottom: 20px;
}

.price-label {
    font-size: 16px;
    color: #44546A;
}

.price-value {
    font-size: 42px;
    font-weight: 700;
    color: #0B3D70;
}

/* Information cards */
.info-card {
    background: #F7F9FC;
    border: 1px solid #E1E7EF;
    padding: 20px;
    border-radius: 10px;
    height: 100%;
}

/* Disclaimer */
.disclaimer {
    background: #FFF8E6;
    border-left: 5px solid #E0A800;
    padding: 15px;
    border-radius: 6px;
    margin-top: 25px;
}

/* Footer */
.footer {
    text-align: center;
    color: #667085;
    font-size: 14px;
    padding-top: 20px;
    padding-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open("used_car_price_model.pkl", "rb") as file:
        model = pickle.load(file)

    return model


model = load_model()


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="project-header">

<h1>🚗 Used Car Price Predictor</h1>

<p>
Machine Learning Regression Project |
Tuned Random Forest Model
</p>

<p>
<b>By Phiwokuhle Sibisi</b><br>
Student No: 1519384
</p>

<p>
Enter the vehicle characteristics below to estimate
its predicted selling price.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        label="R² Score",
        value="86.36%"
    )


with col2:

    st.metric(
        label="Mean Absolute Error",
        value="R69,610"
    )


with col3:

    st.metric(
        label="Model",
        value="Random Forest"
    )


st.caption(
    "Performance values are based on the held-out test dataset "
    "used during model evaluation."
)

st.divider()


# =========================================================
# VEHICLE INPUT
# =========================================================

st.subheader("Vehicle Details")

st.write(
    "Enter the four vehicle characteristics used by the "
    "trained machine-learning model."
)


left, right = st.columns(2)


# ---------------------------------------------------------
# LEFT COLUMN
# ---------------------------------------------------------

with left:

    max_power = st.number_input(
        "Maximum Power (bhp)",
        min_value=1.0,
        max_value=500.0,
        value=100.0,
        step=1.0,
        help="The maximum power produced by the vehicle's engine."
    )


    engine = st.number_input(
        "Engine Size (CC)",
        min_value=500,
        max_value=6000,
        value=1500,
        step=100,
        help="Engine displacement measured in cubic centimetres."
    )


# ---------------------------------------------------------
# RIGHT COLUMN
# ---------------------------------------------------------

with right:

    vehicle_age = st.number_input(
        "Vehicle Age (Years)",
        min_value=0,
        max_value=30,
        value=5,
        step=1,
        help="The age of the vehicle in years."
    )


    km_driven = st.number_input(
        "Kilometres Driven",
        min_value=0,
        max_value=500000,
        value=60000,
        step=1000,
        help="Total kilometres travelled by the vehicle."
    )


st.write("")


# =========================================================
# PREDICTION BUTTON
# =========================================================

predict_button = st.button(
    "Calculate Predicted Selling Price",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # CREATE INPUT DATA
    # -----------------------------------------------------

    # The column names and order must match the features
    # used when training the Random Forest model.

    input_data = pd.DataFrame({
        "max_power": [max_power],
        "engine": [engine],
        "vehicle_age": [vehicle_age],
        "km_driven": [km_driven]
    })


    # -----------------------------------------------------
    # GENERATE PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(input_data)

    predicted_price = prediction[0]


    # -----------------------------------------------------
    # DISPLAY PRICE
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div class="price-card">

            <div class="price-label">
                ESTIMATED SELLING PRICE
            </div>

            <div class="price-value">
                R {predicted_price:,.2f}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # VEHICLE SUMMARY
    # -----------------------------------------------------

    st.subheader("Prediction Summary")

    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Maximum Power",
        f"{max_power:,.0f} bhp"
    )


    c2.metric(
        "Engine Size",
        f"{engine:,} CC"
    )


    c3.metric(
        "Vehicle Age",
        f"{vehicle_age} years"
    )


    c4.metric(
        "KM Driven",
        f"{km_driven:,} km"
    )


# =========================================================
# FEATURE INFORMATION
# =========================================================

st.divider()

st.subheader("Features Used by the Model")

st.write(
    """
    The final model uses the four features selected during
    the feature-selection stage of the project.
    """
)


c1, c2 = st.columns(2)


# ---------------------------------------------------------
# FEATURE CARD 1
# ---------------------------------------------------------

with c1:

    st.markdown("""
    <div class="info-card">

    <b>Maximum Power</b><br>
    Represents the maximum power output of the vehicle's engine.

    <br><br>

    <b>Engine Size</b><br>
    Represents engine displacement measured in cubic
    centimetres (CC).

    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# FEATURE CARD 2
# ---------------------------------------------------------

with c2:

    st.markdown("""
    <div class="info-card">

    <b>Vehicle Age</b><br>
    Represents the age of the vehicle in years.

    <br><br>

    <b>Kilometres Driven</b><br>
    Represents the total distance travelled by the vehicle.

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader("About the Machine Learning Model")

st.write(
    """
    The application uses the **Tuned Random Forest Regressor**
    selected during the model-evaluation stage of the project.

    The model was trained using four selected features:
    **maximum power, engine size, vehicle age and kilometres
    driven**.

    During testing, the model achieved an **R² score of 0.8636**
    and a **Mean Absolute Error (MAE) of approximately R69,610**.

    The R² result means that the model explained approximately
    **86.36% of the variation in selling prices within the
    test dataset**.
    """
)


# =========================================================
# MODEL FEATURE IMPORTANCE
# =========================================================

st.subheader("Model Feature Importance")

st.write(
    """
    The Random Forest feature-importance analysis showed that
    the four selected variables contributed differently to
    the model's predictions.
    """
)


importance_data = pd.DataFrame({
    "Feature": [
        "Maximum Power",
        "Vehicle Age",
        "Engine",
        "KM Driven"
    ],

    "Importance": [
        0.500616,
        0.343268,
        0.123665,
        0.032452
    ]
})


st.bar_chart(
    importance_data.set_index("Feature")
)


st.caption(
    "Maximum power had the highest feature importance, "
    "followed by vehicle age, engine size and kilometres driven."
)


# =========================================================
# MODEL WORKFLOW
# =========================================================

st.divider()

st.subheader("Prediction Workflow")

st.write(
    """
    **1.** The user enters the vehicle characteristics.

    **2.** Streamlit captures the values.

    **3.** Pandas creates a DataFrame containing the four
    model features.

    **4.** The saved Tuned Random Forest model processes
    the new observation.

    **5.** The predicted selling price is displayed in
    South African Rand (R).
    """
)


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("""
<div class="disclaimer">

<b>Academic Project Disclaimer</b><br><br>

This prediction is generated by a machine-learning model developed
for academic purposes.

The predicted value is an estimate based on patterns learned from
the project dataset and should not be regarded as an official
vehicle valuation, dealership quotation or financial advice.

</div>
""", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.write("")

st.markdown("""
<div class="footer">

<b>Exploratory Data Analysis & Machine Learning Major Project</b>
<br>

Used Car Price Prediction
<br><br>

<b>Phiwokuhle Sibisi</b>
<br>

Student No: 1519384

</div>
""", unsafe_allow_html=True)