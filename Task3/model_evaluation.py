import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Create dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score': [35, 45, 48, 52, 68, 68, 75, 82, 88, 95]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Hours']]
y = df['Score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f} points")

# Compare actual vs predicted
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print("\nActual vs Predicted Scores:")
print(results)

# Predict for a new student
new_hours = pd.DataFrame({'Hours': [4.5]})
predicted_score = model.predict(new_hours)

print(f"\nPredicted score for 4.5 study hours: {predicted_score[0]:.2f}")