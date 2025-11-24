import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc
import numpy as np

# Load dataset
file_path = '/Users/arinayare/QR JPMC /Loan Example(task 3&4).csv'
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop(columns=['customer_id', 'default'])
y = data['default']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
logreg = LogisticRegression(max_iter=200, class_weight='balanced')
logreg.fit(X_train_scaled, y_train)
probs_logreg = logreg.predict_proba(X_test_scaled)[:, 1]

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
probs_rf = rf.predict_proba(X_test)[:, 1]

# Model evaluation function
def evaluate_model(y_true, probs, name):
    auc_score = roc_auc_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    print(f"{name} -> AUC: {auc_score:.4f}, Brier: {brier:.4f}, PR AUC: {pr_auc:.4f}")

# Evaluate both models
evaluate_model(y_test, probs_logreg, "Logistic Regression")
evaluate_model(y_test, probs_rf, "Random Forest")

# Function to calculate expected loss
def expected_loss(loan_features, model, scaler=None, recovery_rate=0.10):
    df = pd.DataFrame([loan_features])
    df_scaled = scaler.transform(df) if scaler else df
    pd_est = model.predict_proba(df_scaled)[:, 1][0]   # Probability of default
    ead = df['loan_amt_outstanding'].values[0]         # Exposure at Default
    lgd = 1 - recovery_rate                            # Loss Given Default
    expected_loss = pd_est * ead * lgd
    return {'prob_default': pd_est, 'expected_loss': expected_loss}

# Test expected_loss function on one record
example_loan = X_test.iloc[0].to_dict()
result = expected_loss(example_loan, logreg, scaler)

print("\nExample Loan Prediction:")
print(result)
