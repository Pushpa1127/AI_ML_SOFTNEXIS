from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load Iris dataset
iris = load_iris()

# Features and labels
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# Scale features
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train k-NN model on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=3)
knn_scaled.fit(X_train_scaled, y_train)

# Predict
y_pred_scaled = knn_scaled.predict(X_test_scaled)

# Accuracy
new_accuracy = knn_scaled.score(X_test_scaled, y_test)

print(f"Scaled Test Accuracy: {new_accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_scaled,
    target_names=iris.target_names
))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_scaled))