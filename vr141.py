import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Prepare features and target
X = train_df.drop(columns=["id", "efficiency"])
y = train_df["efficiency"]
X_test = test_df.drop(columns=["id"])

# Column types
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# Preprocessors
numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Final model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42))
])

# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"Validation RMSE: {rmse:.4f}")

# Predict on test data
test_preds = model.predict(X_test)

# Prepare submission
submission = pd.DataFrame({
    "id": test_df["id"],
    "efficiency": test_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
