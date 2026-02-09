import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.tree import plot_tree

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
rf=RandomForestClassifier(n_estimators=100,max_depth=7,random_state=42,n_jobs=-1)
logo =LeaveOneGroupOut()

fold_accuracies=[]
all_y_true=[]
all_y_pred=[]

print(f'starting LOSO cross-validation on {len(X.columns)} features...')
for train_idx, test_idx in logo.split(X,y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train
    rf.fit(X_train,y_train)

    # Predict
    preds = rf.predict(X_test)
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

