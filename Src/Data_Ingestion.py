
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, silhouette_score, mean_absolute_error

# ==================================================
# PATH CONFIGURATION
# ==================================================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH_ACCEPT = os.path.join(BASE_DIR, "..","Data", "accepted_2007_to_2018Q4.csv")
DATA_PATH_REJECT = os.path.join(BASE_DIR, "..","Data", "rejected_2007_to_2018Q4.csv")
DATA_PATH_FINAL = os.path.join(BASE_DIR, "..","Data", "CombinedLoanDataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..","Model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================
def LoadDataset_Filter():
    # Load Accepted Data
    acc = pd.read_csv(DATA_PATH_ACCEPT, low_memory=False)
    acc['issue_d'] = pd.to_datetime(acc['issue_d'])
    acc = acc[(acc['issue_d'].dt.year >= 2016) & (acc['issue_d'].dt.year <= 2018)]

    # Load Rejected Data
    rej = pd.read_csv(DATA_PATH_REJECT)
    rej['Application Date'] = pd.to_datetime(rej['Application Date'])
    rej = rej[(rej['Application Date'].dt.year >= 2016) & (rej['Application Date'].dt.year <= 2018)]

    return acc, rej

# ===================================================
# COMBINE DATASET, CLEAN AND SAVE
# ====================================================
def Combined_Cleaned_Dataset(acc, rej):
    # process Rejected
    rej_clean = pd.DataFrame({
        'amount': rej['Amount Requested'],
        'risk_score': rej['Risk_Score'],
        'dti': rej['Debt-To-Income Ratio'].str.replace('%', '').astype(float),
        'state': rej['State'],
        'emp_length': rej['Employment Length'],
        'accepted': 0
    })

    # process Accepted
    acc_clean = pd.DataFrame({
        'amount': acc['loan_amnt'],
        'risk_score': (acc['fico_range_high'] + acc['fico_range_low']) / 2,
        'dti': acc['dti'],
        'state': acc['addr_state'],
        'emp_length': acc['emp_length'],
        'accepted': 1
    })
    
    #Combine together and remove fully empty rows => dropna
    concatenatedData = pd.concat([acc_clean, rej_clean], axis=0).dropna()
    # Remove duplicates
    concatenatedData.drop_duplicates(inplace=True)
    return concatenatedData