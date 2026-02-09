import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler


# Load data
df=pd.read_csv('/wesad_global_features.csv')
# Data Cleaning
# Drop columns that are 100% empty and fill minor NaNs with the median
df=df.dropna(axis=1,how='all')
df=df.fillna(df.median(numeric_only=True))
# extract X(features), y(label), and groups(subjects)
X=df.drop(['Subject_ID','Label'],axis=1)
y=df['Label']
groups=df['Subject_ID']
# Setup the model and loso , max_width=7
xgb = XGBClassifier(n_estimators=100, learning_rate=0.05,
                         max_depth=6,
                         eval_metric='logloss', random_state=42)
logo =LeaveOneGroupOut()

fold_accuracies=[]
all_y_true=[]
all_y_pred=[]

print(f'starting LOSO cross-validation on {len(X.columns)} features...')
for train_idx, test_idx in logo.split(X,y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Standardising the data 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    xgb.fit(X_train,y_train)

    # Predict
    preds = xgb.predict(X_test)
    accuracy = accuracy_score(y_test,preds)

    fold_accuracies.append(accuracy)
    all_y_true.extend(y_test)
    all_y_pred.extend(preds)

    current_sub = groups.iloc[test_idx[0]]
    print(f'Subject {current_sub}: accuracy = {accuracy:.2%}')

# Final Results
print("\n" + "="*30)
print(f"OVERALL LOSO ACCURACY: {np.mean(fold_accuracies):.2%}")
print("="*30)
print(classification_report(all_y_true, all_y_pred, target_names=['Baseline','Stress']))

from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from xgboost import XGBClassifier

# Use the Best Settings found as a starting point, but refine them
param_grid = {
    'max_depth': [6, 7, 8],
    'learning_rate': [0.05], # We know this works well
    'n_estimators': [200, 300],
    'subsample': [0.8],
    'colsample_bytree': [0.8, 1.0] # New parameter to prevent overfitting
}

# Use LeaveOneGroupOut so the grid search respects the different humans
logo = LeaveOneGroupOut()

grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric='logloss'),
    param_grid=param_grid,
    cv=logo, # <--- THIS IS THE KEY: Tune based on new subjects, not random rows
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("Starting LOSO-based Grid Search...")
grid_search.fit(X, y, groups=groups) 

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

# 1. Use cross_val_predict to get "out-of-fold" predictions for every subject
# This uses the best estimator found by GridSearchCV
print("\n Generating cross-validated predictions using the tuned model...")
y_pred_tuned = cross_val_predict(
    grid_search.best_estimator_, 
    X, y, 
    groups=groups, 
    cv=logo, 
    n_jobs=-1
)

# 2. Print the final Classification Report
print("\n" + "="*30)
print(f" TUNED LOSO ACCURACY: {grid_search.best_score_:.2%}")
print("="*30)
print(classification_report(y, y_pred_tuned, target_names=['Baseline', 'Stress']))


print(f" True Best Settings: {grid_search.best_params_}")
print(f" True Best LOSO Score: {grid_search.best_score_:.2%}")

