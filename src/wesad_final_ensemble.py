from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
import seaborn as sns



# Load data
df=pd.read_csv('/home/Linux1/python/pythonfiles/wesad_global_features.csv')
# Data Cleaning
# Drop columns that are 100% empty and fill minor NaNs with the median
df=df.dropna(axis=1,how='all')
df=df.fillna(df.median(numeric_only=True))
# extract X(features), y(label), and groups(subjects)
X=df.drop(['Subject_ID','Label'],axis=1)
y=df['Label']
groups=df['Subject_ID']

all_y_true = []
all_y_pred = []

# Define your individual models in a dictionary
individual_models = {
    "RF_Baseline": RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42),
    "SVM_Baseline": SVC(kernel='rbf', probability=True, random_state=42), # probability=True is required for soft voting
    "XGB_Tuned": XGBClassifier(
        colsample_bytree=1.0, 
        learning_rate=0.05, 
        max_depth=8, 
        n_estimators=200, 
        subsample=0.8,
        eval_metric='logloss'
    )
}

# Setup the Voting Ensemble
# We convert the dictionary into a list of (name, model) tuples for Scikit-Learn
ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in individual_models.items()],
    voting='soft' # 'soft' uses probabilities and usually works better for biosensors
)

# Initialize your subject-wise results storage
# We'll use a list of dicts to easily convert to a DataFrame later
subject_results = []

logo = LeaveOneGroupOut()
groups = df['Subject_ID'] # Assuming 'subject' column holds S2, S3, etc.

# Add the Ensemble to our testing list
all_to_test = {**individual_models, "FINAL_ENSEMBLE": ensemble}

print(" Starting LOSO Evaluation...")

# Corrected LOSO Loop (One loop through subjects only)
for train_idx, test_idx in logo.split(X, y, groups=groups):
    # Identify which subject is being tested in this fold
    current_subject = groups.iloc[test_idx].unique()[0]
    print(f"Processing Subject {current_subject}...")
    
    # Split data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Scale data (crucial for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define a dictionary for THIS subject's scores
    fold_scores = {'Subject': current_subject}
    
    # Fit and Predict every model on THIS specific subject
    for name, model in all_to_test.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        fold_scores[name] = acc
    
    # Append the result for this subject (1 row per subject)
    subject_results.append(fold_scores)

# Inside the loop...
# Fit the ensemble on the training data
ensemble.fit(X_train_scaled, y_train)

# Get predictions for the current test subject
y_ensemble_pred = ensemble.predict(X_test_scaled)

# Store these predictions and the true labels
all_y_true.extend(y_test)
all_y_pred.extend(y_ensemble_pred)

# Create the Final "Performance Matrix"
df_results = pd.DataFrame(subject_results)

# Calculate Average accurately
avg_row = df_results.mean(numeric_only=True)
df_results.loc['Average'] = avg_row
df_results.at['Average', 'Subject'] = 'MEAN'

print("\n" + "="*50)
print(df_results)

# Save the detailed subject-wise matrix
df_results.to_csv('wesad_final_loso_results.csv', index=False)

# Save your winning Ensemble model for future use
import joblib
joblib.dump(ensemble, 'final_stress_ensemble_model.joblib')

print("✅ Results and Model saved successfully!")

# Extract the Mean Ensemble Accuracy
# We use .at because we know exactly where it is (Row 'Average', Column 'FINAL_ENSEMBLE')
final_acc = df_results.at['Average', 'FINAL_ENSEMBLE']

print("\n" + "*"*50)
print(f" FINAL ENSEMBLE ACCURACY: {final_acc:.2%}")
print("*"*50 + "\n")

from sklearn.metrics import classification_report

print("\n" + "="*20 + " FINAL GLOBAL PERFORMANCE " + "="*20)
# Generate the report using the collected lists
report = classification_report(all_y_true, all_y_pred, target_names=['Baseline', 'Stress'])

print(report)

