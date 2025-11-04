import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
import os

np.random.seed(42)

# Generate synthetic dataset
n = 300
df = pd.DataFrame({
    "attendance": np.random.randint(50, 100, n),
    "assignments": np.random.randint(40, 100, n),
    "internals": np.random.randint(30, 100, n),
    "extra_activities": np.random.randint(0, 10, n),
    "previous_gpa": np.round(np.random.uniform(4, 10, n), 2)
})

# Dropout probability (logistic style)
att_scaled = (65 - df["attendance"]) / 20.0
int_scaled = (45 - df["internals"]) / 15.0
base_p = 1 / (1 + np.exp(- (att_scaled + int_scaled + np.random.normal(0, 0.5, n))))
df["dropout_prob"] = base_p
df["dropout"] = (np.random.rand(n) < base_p).astype(int)

# Grade calculation
df["grade"] = np.clip(
    0.5 * df["previous_gpa"] +
    0.2 * (df["assignments"] / 10) +
    0.3 * (df["internals"] / 10) +
    np.random.normal(0, 0.5, n),
    0, 10
)

X = df[["attendance", "assignments", "internals", "extra_activities", "previous_gpa"]]
y_class = df["dropout"]
y_reg = df["grade"]

X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train2, X_test2, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Logistic Regression for dropout
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train_c)
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:, 1]
print("Dropout classifier accuracy:", accuracy_score(y_test_c, preds))
print("Sample predicted dropout probabilities:", probs[:10])

# Random Forest for grade
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train2, y_train_r)
preds_r = reg.predict(X_test2)
print("Grade regressor MAE:", mean_absolute_error(y_test_r, preds_r))

# Save models
os.makedirs("static", exist_ok=True)
joblib.dump(clf, "static/dropout_model.pkl")
joblib.dump(reg, "static/grade_model.pkl")
print("Models saved in static/")
df.to_csv("static/sample_data.csv", index=False)
