import pandas as pd

# Load the dataset
data = pd.read_csv("Data/Student_Performance.csv")

# Display the first 5 rows
print(data.head())
# Check the number of rows and columns
print("\nDataset Shape:")
print(data.shape)

# Check column names
print("\nColumn Names:")
print(data.columns)

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())
# Check data typespython student_performance.py
print("\nData Types:")
print(data.dtypes)
# Check data types
print("\nData Types:")
print(data.dtypes)
# Convert Yes/No into numerical values
data["Extracurricular Activities"] = data["Extracurricular Activities"].map({"Yes": 1, "No": 0})

print("\nAfter Encoding:")
print(data.head())
# Separate features and target
X = data.drop("Performance Index", axis=1)
y = data["Performance Index"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)
from sklearn.linear_model import LinearRegression

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")
# Make predictions on the test data
y_pred = model.predict(X_test)

print("\nFirst 5 Predictions:")
print(y_pred[:5])
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
from sklearn.tree import DecisionTreeRegressor

# Create the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")
# Make predictions using Decision Tree
dt_pred = dt_model.predict(X_test)

print("\nFirst 5 Decision Tree Predictions:")
print(dt_pred[:5])
# Evaluate the Decision Tree model
dt_mae = mean_absolute_error(y_test, dt_pred)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
dt_r2 = r2_score(y_test, dt_pred)

print("\nDecision Tree Evaluation:")
print("MAE:", dt_mae)
print("RMSE:", dt_rmse)
print("R2 Score:", dt_r2)
from sklearn.ensemble import RandomForestRegressor

# Create the Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")
# Make predictions using Random Forest
rf_pred = rf_model.predict(X_test)

print("\nFirst 5 Random Forest Predictions:")
print(rf_pred[:5])
# Evaluate the Random Forest model
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Evaluation:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)
from sklearn.svm import SVR

# Create the SVR model
svr_model = SVR()

# Train the model
svr_model.fit(X_train, y_train)

print("\nSVR model trained successfully!")
# Make predictions using SVR
svr_pred = svr_model.predict(X_test)

print("\nFirst 5 SVR Predictions:")
print(svr_pred[:5])
# Evaluate the SVR model
svr_mae = mean_absolute_error(y_test, svr_pred)
svr_rmse = np.sqrt(mean_squared_error(y_test, svr_pred))
svr_r2 = r2_score(y_test, svr_pred)

print("\nSVR Evaluation:")
print("MAE:", svr_mae)
print("RMSE:", svr_rmse)
print("R2 Score:", svr_r2)
# Compare all models
results = {
    "Linear Regression": r2,
    "Decision Tree": dt_r2,
    "Random Forest": rf_r2,
    "SVR": svr_r2
}

best_model_name = max(results, key=results.get)

print("\nBest Model:")
print(best_model_name)
# Predict performance for a new student
# Select the best model automatically
models = {
    "Linear Regression": model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "SVR": svr_model
}

best_model = models[best_model_name]

print("\nUsing Best Model:", best_model_name)

# Predict performance for a new student
hours_studied = float(input("\nEnter Hours Studied: "))
previous_scores = float(input("Enter Previous Scores: "))
extracurricular = int(input("Extracurricular Activities (Yes=1, No=0): "))
sleep_hours = float(input("Enter Sleep Hours: "))
sample_papers = float(input("Enter Sample Question Papers Practiced: "))

new_student = pd.DataFrame([[
    hours_studied,
    previous_scores,
    extracurricular,
    sleep_hours,
    sample_papers
]], columns=X.columns)

prediction = best_model.predict(new_student)

print("\nPredicted Performance Index:", prediction[0])
# Display feature coefficients
print("\nFeature Coefficients:")

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)
    # Final Model Comparison

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "SVR"
    ],
    "MAE": [
        mae,
        dt_mae,
        rf_mae,
        svr_mae
    ],
    "RMSE": [
        rmse,
        dt_rmse,
        rf_rmse,
        svr_rmse
    ],
    "R2 Score": [
        r2,
        dt_r2,
        rf_r2,
        svr_r2
    ]
})

print("\nFinal Model Comparison:")
print(comparison.round(4).to_string(index=False))
import matplotlib.pyplot as plt

# Model comparison graph
models = ["Linear Regression", "Decision Tree", "Random Forest", "SVR"]
r2_scores = [r2, dt_r2, rf_r2, svr_r2]

plt.figure(figsize=(8, 5))
plt.bar(models, r2_scores)

plt.title("Model Comparison - R² Score")
plt.xlabel("Machine Learning Models")
plt.ylabel("R² Score")
plt.ylim(0.95, 1.0)

plt.xticks(rotation=15)
plt.tight_layout()

plt.show()
# MAE Comparison Graph
plt.figure(figsize=(8, 5))
plt.bar(models, [mae, dt_mae, rf_mae, svr_mae])

plt.title("Model Comparison - MAE")
plt.xlabel("Machine Learning Models")
plt.ylabel("Mean Absolute Error")

plt.xticks(rotation=15)
plt.tight_layout()

plt.show()


# RMSE Comparison Graph
plt.figure(figsize=(8, 5))
plt.bar(models, [rmse, dt_rmse, rf_rmse, svr_rmse])

plt.title("Model Comparison - RMSE")
plt.xlabel("Machine Learning Models")
plt.ylabel("Root Mean Squared Error")

plt.xticks(rotation=15)
plt.tight_layout()

plt.show()