import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Analyze Risk Factor: FICO/Risk Score vs Acceptance
def HistPlotsForAnalysis(dataset):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=dataset, x='risk_score', hue='accepted', bins=50, kde=True)
    plt.title('Risk Score Distribution: Accepted vs Rejected')
    #plt.show() //No need to show after analysis
    return

def AcceptedLoanBasedOnDate_Plot(accepted):
    # Analyze Loan Performance over time (Accepted Loans only)
    accepted['month_year'] = accepted['issue_d'].dt.to_period('M')
    accepted.groupby('month_year')['loan_amnt'].sum().plot(kind='line', figsize=(12,5))
    plt.title('Total Loan Volume Trend (2016-2018)')
    plt.ylabel('Amount ($)')
    #plt.show()
