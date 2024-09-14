
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset (Boston housing dataset is deprecated, use California housing)
from sklearn.datasets import fetch_california_housing

# Load California housing data
data = fetch_california_housing()

# Features (X) and target (y)
X = data.data
y = data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model and the scaler for later use
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model training completed and saved!")
