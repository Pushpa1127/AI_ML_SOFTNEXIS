from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = load_iris()

# Features and labels
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,      # 30% test data
    random_state=42
)

# Display information
print("Total samples:", len(X))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nFeature shape:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("\nLabel shape:")
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)