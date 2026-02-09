import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, classification_report
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
svm =  SVC(kernel='rbf', random_state=42)

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
    svm.fit(X_train_scaled,y_train)

    # Predict
    preds = svm.predict(X_test_scaled)
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

