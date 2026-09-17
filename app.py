import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load dataset
data = pd.read_csv("Data/Student_Performance.csv")

# Encode extracurricular activities
data["Extracurricular Activities"] = data["Extracurricular Activities"].map({
    "Yes": 1,
    "No": 0
})

# Separate features and target
X = data.drop("Performance Index", axis=1)
y = data["Performance Index"]

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)
# Model Performance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

evaluation_model = LinearRegression()
evaluation_model.fit(X_train, y_train)

y_pred = evaluation_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Title
st.sidebar.header("🤖 Model Information")
st.sidebar.write("Best Model: Linear Regression")

st.sidebar.header("📊 Model Performance")

st.sidebar.metric("R² Score", f"{r2:.4f}")
st.sidebar.metric("MAE", f"{mae:.4f}")
st.sidebar.metric("RMSE", f"{rmse:.4f}")


st.write(
    "A Machine Learning application that predicts a student's "
    "Performance Index using academic and lifestyle-related factors."
)
st.info(
    "Enter the student's details below and click Predict Performance "
    "to generate the predicted Performance Index."
)

# Input section
st.subheader("Enter Student Details")
st.caption("Enter the student's academic and lifestyle details below.")

hours_studied = st.number_input(
    "📚 Hours Studied",
    min_value=0,
    max_value=24,
    value=7
)

previous_scores = st.number_input(
    "📝 Previous Scores",
    min_value=0,
    max_value=100,
    value=80
)

extracurricular = st.selectbox(
    "🎯 Extracurricular Activities",
    ["Yes", "No"]
)

sleep_hours = st.number_input(
    "😴 Sleep Hours",
    min_value=0,
    max_value=24,
    value=7
)

sample_papers = st.number_input(
    "📄 Sample Question Papers Practiced",
    min_value=0,
    max_value=20,
    value=5
)

# Prediction
if st.button("🔮 Predict Performance"):

    extracurricular_value = 1 if extracurricular == "Yes" else 0

    new_student = pd.DataFrame([[
        hours_studied,
        previous_scores,
        extracurricular_value,
        sleep_hours,
        sample_papers
    ]], columns=X.columns)

    prediction = model.predict(new_student)[0]

    st.subheader("📊 Prediction Result")

st.success(
    f"Predicted Performance Index: {prediction:.2f}"
)

if prediction >= 75:
    st.info("Performance Level: High")
elif prediction >= 50:
    st.info("Performance Level: Moderate")
else:
    st.info("Performance Level: Needs Improvement")(
        f"📊 Predicted Performance Index: {prediction:.2f}"
    )
    st.divider()

st.subheader("📈 Model Comparison")

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "SVR"
    ],
    "R² Score": [
        0.9890,
        0.9762,
        0.9861,
        0.9855
    ]
})

st.bar_chart(
    comparison.set_index("Model")
)
st.divider()

st.subheader("📌 Prediction Factors")

st.write("""
The model uses the following factors to predict student performance:
""")

st.markdown("""
- 📚 **Hours Studied**
- 📝 **Previous Scores**
- 🎯 **Extracurricular Activities**
- 😴 **Sleep Hours**
- 📄 **Sample Question Papers Practiced**
""")