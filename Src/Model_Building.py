import os
import pickle

import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from Data_Ingestion import LoadDataset_Filter, Combined_Cleaned_Dataset, BASE_DIR,DATA_PATH_ACCEPT,DATA_PATH_REJECT, DATA_PATH_FINAL, MODEL_DIR
from Analysis_EDA import HistPlotsForAnalysis, AcceptedLoanBasedOnDate_Plot
# ==================================================
# LOAD DATA
# ==================================================
acc, rej = LoadDataset_Filter()

# ===================================================
# COMBINE DATASET, CLEAN AND SAVE
# ====================================================

concatenatedData = Combined_Cleaned_Dataset(acc,rej)
# Save the cleaned dataset
concatenatedData.to_csv(DATA_PATH_FINAL, index=False)

#EDA
HistPlotsForAnalysis(concatenatedData)
#AcceptedLoanBasedOnDate_Plot(concatenatedData)

# ==================================================
# ENDODE CATEGORICAL FEATURE
# ==================================================
lableEncoder = LabelEncoder()
concatenatedData['state'] = lableEncoder.fit_transform(concatenatedData['state'])
concatenatedData['emp_length'] = lableEncoder.fit_transform(concatenatedData['emp_length'].astype(str))

# ===================================================
# SPLIT FEATURES & TARGET
# ===================================================
X = concatenatedData.drop('accepted', axis=1)
y = concatenatedData['accepted']

# Save feature names
#joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "feature_names.pkl"))
print("Feature names Identified")

# ===================================================
# SCALING
# ===================================================
scaler = StandardScaler() # mean - 0, #std - 1
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X)

# ===================================================
# TRAIN_TEST SPLIT
# ===================================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


#joblib.dump(scaler,os.path.join(MODEL_DIR,"scaler.pkl"))
print("Scaler info Done")

# ===================================================
# MODEL 1 - XGBOOST
# ===================================================
xgb_Model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state =42)
xgb_Model.fit(X_train, y_train)
print("XGBClassifier  Done")

# Predicting 'amount' for people likely to be accepted
Data_accept_only = concatenatedData[concatenatedData['accepted'] == 1]
X_reg = Data_accept_only[['risk_score', 'dti', 'state', 'emp_length']]
y_reg = Data_accept_only['amount'] # predict continuous

reg = XGBRegressor()
reg.fit(X_reg, y_reg)
print("XGBRegressor  Done")

# ===================================================
# CLUSTURING
# ===================================================
kmeans = KMeans(n_clusters=4, random_state=42)
concatenatedData['segment'] = kmeans.fit_predict(X_scaled)
print("kmeans clustering  Done")

# Classification Eval
y_pred = xgb_Model.predict(X_test)
print(classification_report(y_test, y_pred))

# SAVE the model
state_mapping = {label: i for i, label in enumerate(lableEncoder.classes_)}
default_state_val = int(len(lableEncoder.classes_) / 2) # A middle-ground index

artifacts = {
    'classifier': xgb_Model,
    'regressor': reg,
    'scaler': scaler,
    'state_mapping': state_mapping, # Save the dict instead of the LE object
    'default_state': default_state_val
}


with open(os.path.join(MODEL_DIR,"lending_club_pipeline.pkl"), 'wb') as f:
    pickle.dump(artifacts, f)

print("All models Saved Successfully.")