import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression         
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# --- 2. Load Data ---
df = pd.read_csv("data.csv")
print("Dataset shape:", df.shape)

# --- 3. Preprocessing ---
# Drop irrelevant columns
df = df.drop(["date", "street"], axis=1, errors="ignore")

# Handle missing values
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Remove outliers (3-sigma rule on price & sqft_living)
def remove_outliers(data, col):
    mean, std = data[col].mean(), data[col].std()
    return data[(data[col] >= mean - 3*std) & (data[col] <= mean + 3*std)]

df = remove_outliers(df, "price")
df = remove_outliers(df, "sqft_living")

# Feature engineering: house age
df["age"] = np.where(df["yr_renovated"] > 0,
                     2025 - df["yr_renovated"],
                     2025 - df["yr_built"])
df = df.drop(["yr_built", "yr_renovated"], axis=1, errors="ignore")

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# --- 4. Split Features and Target ---
X = df.drop("price", axis=1)
y = df["price"]

# Keep only numeric columns (safety)
X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 5. Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --- 6. Train the Built-in Linear Regression Model ---
model = LinearRegression()          # <-- This is the built-in model
model.fit(X_train_scaled, y_train)

# --- 7. Predictions ---
y_train_pred = model.predict(X_train_scaled)
y_test_pred  = model.predict(X_test_scaled)

# --- 8. Evaluation ---
def report(name, y_true, y_pred):
    print(f"\n{name} Performance")
    print(f"  MSE : {mean_squared_error(y_true, y_pred):,.2f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):,.2f}")
    print(f"  MAE : {mean_absolute_error(y_true, y_pred):,.2f}")
    print(f"  R²  : {r2_score(y_true, y_pred):.4f}")

report("Training", y_train, y_train_pred)
report("Test",     y_test,  y_test_pred)

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train,
                            cv=5, scoring="r2")
print(f"\n5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# --- 9. Visualizations ---
plt.figure(figsize=(12, 4))

# Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted (R² = {r2_score(y_test, y_test_pred):.4f})")

# Residuals
plt.subplot(1, 2, 2)
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(0, color="r", linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.tight_layout()
plt.savefig("model_results.png", dpi=100)
plt.show()

# --- 10. Top 10 Most Influential Features ---
coefs = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Positive Coefficients:")
print(coefs.head(10))
print("\nTop 10 Negative Coefficients:")
print(coefs.tail(10))

print(f"\nIntercept: {model.intercept_:,.2f}")