"""
Hero FinCorp Case Study

This file performs all 20 actions one by one using below libraries:
pandas, numpy, matplotlib, and seaborn.


All charts will be saved inside the folder: beginner_outputs
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# BASIC SETUP
# =============================================================================

DATA_PATH = r"C:\Users\auchi\Downloads\Datasets_hero_fin"
OUTPUT_PATH = "chart_outputs"

os.makedirs(OUTPUT_PATH, exist_ok=True)
sns.set(style="whitegrid")


# =============================================================================
# LOAD DATASETS
# =============================================================================

customers = pd.read_csv(os.path.join(DATA_PATH, "customers.csv"))
loans = pd.read_csv(os.path.join(DATA_PATH, "loans.csv"))
applications = pd.read_csv(os.path.join(DATA_PATH, "applications.csv"))
transactions = pd.read_csv(os.path.join(DATA_PATH, "transactions.csv"))
defaults = pd.read_csv(os.path.join(DATA_PATH, "defaults.csv"))
branches = pd.read_csv(os.path.join(DATA_PATH, "branches.csv"))

print("All datasets loaded successfully.")
print("Customers:", customers.shape)
print("Loans:", loans.shape)
print("Applications:", applications.shape)
print("Transactions:", transactions.shape)
print("Defaults:", defaults.shape)
print("Branches:", branches.shape)


# =============================================================================
# SIMPLE DATA PREPARATION USED BY MULTIPLE TASKS
# =============================================================================

# Convert date columns to proper date format
applications["Application_Date"] = pd.to_datetime(applications["Application_Date"], errors="coerce")
applications["Approval_Date"] = pd.to_datetime(applications["Approval_Date"], errors="coerce")

loans["Disbursal_Date"] = pd.to_datetime(loans["Disbursal_Date"], errors="coerce")
loans["Repayment_Start_Date"] = pd.to_datetime(loans["Repayment_Start_Date"], errors="coerce")
loans["Repayment_End_Date"] = pd.to_datetime(loans["Repayment_End_Date"], errors="coerce")

transactions["Transaction_Date"] = pd.to_datetime(transactions["Transaction_Date"], errors="coerce")
defaults["Default_Date"] = pd.to_datetime(defaults["Default_Date"], errors="coerce")

# Create default flag using Loan_ID
defaults_small = defaults[["Loan_ID", "Default_Amount", "Recovery_Amount", "Default_Date", "Default_Reason", "Legal_Action"]]
defaults_small["Default_Flag"] = 1

# Merge loan and customer data
loan_data = loans.merge(customers, on="Customer_ID", how="left")

# Add default information
loan_data = loan_data.merge(defaults_small, on="Loan_ID", how="left")
loan_data["Default_Flag"] = loan_data["Default_Flag"].fillna(0)
loan_data["Default_Amount"] = loan_data["Default_Amount"].fillna(0)
loan_data["Recovery_Amount"] = loan_data["Recovery_Amount"].fillna(0)

# Add application information
app_small = applications[["Loan_ID", "Loan_Purpose", "Source_Channel", "Approval_Status", "Processing_Fee", "Application_Date", "Approval_Date"]]
loan_data = loan_data.merge(app_small, on="Loan_ID", how="left")

# Create useful columns
loan_data["Recovery_Rate"] = loan_data["Recovery_Amount"] / loan_data["Default_Amount"].replace(0, np.nan)
loan_data["Recovery_Rate"] = loan_data["Recovery_Rate"].fillna(0)

loan_data["EMI_to_Income"] = (loan_data["EMI_Amount"] * 12) / loan_data["Annual_Income"].replace(0, np.nan)
loan_data["EMI_to_Income"] = loan_data["EMI_to_Income"].fillna(0)

loan_data["Estimated_Interest_Income"] = (
    loan_data["Loan_Amount"] * loan_data["Interest_Rate"] / 100 * loan_data["Loan_Term"] / 12
)

loan_data["Processing_Days"] = (loan_data["Approval_Date"] - loan_data["Application_Date"]).dt.days
loan_data["Time_to_Default_Days"] = (loan_data["Default_Date"] - loan_data["Disbursal_Date"]).dt.days


# =============================================================================
# 1. DATA QUALITY AND PREPARATION
# =============================================================================

print("\n1. DATA QUALITY AND PREPARATION")

datasets = {
    "customers": customers,
    "loans": loans,
    "applications": applications,
    "transactions": transactions,
    "defaults": defaults,
    "branches": branches,
}

for name, data in datasets.items():
    print("\nDataset:", name)
    print("Rows and columns:", data.shape)
    print("Missing values:", data.isnull().sum().sum())
    print("Duplicate rows:", data.duplicated().sum())

# Fill simple missing values
applications["Rejection_Reason"] = applications["Rejection_Reason"].fillna("Not Applicable")
loans["Collateral_Details"] = loans["Collateral_Details"].fillna("Unknown")
defaults["Recovery_Status"] = defaults["Recovery_Status"].fillna("Unknown")


# =============================================================================
# 2. DESCRIPTIVE ANALYSIS
# =============================================================================

print("\n2. DESCRIPTIVE ANALYSIS")
print(loan_data[["Loan_Amount", "EMI_Amount", "Credit_Score"]].describe())

plt.figure(figsize=(8, 5))
sns.histplot(loan_data["Loan_Amount"], bins=30)
plt.title("Distribution of Loan Amount")
plt.savefig(os.path.join(OUTPUT_PATH, "02_loan_amount_distribution.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(loan_data["Credit_Score"], bins=30)
plt.title("Distribution of Credit Score")
plt.savefig(os.path.join(OUTPUT_PATH, "02_credit_score_distribution.png"))
plt.close()


# =============================================================================
# 3. DEFAULT RISK ANALYSIS
# =============================================================================

print("\n3. DEFAULT RISK ANALYSIS")

risk_columns = ["Loan_Amount", "Interest_Rate", "Credit_Score", "EMI_Amount", "Overdue_Amount", "Default_Flag"]
print(loan_data[risk_columns].corr())

plt.figure(figsize=(8, 6))
sns.heatmap(loan_data[risk_columns].corr(), annot=True, cmap="coolwarm")
plt.title("Default Risk Correlation")
plt.savefig(os.path.join(OUTPUT_PATH, "03_default_risk_correlation.png"))
plt.close()


# =============================================================================
# 4. BRANCH AND REGIONAL PERFORMANCE
# =============================================================================

print("\n4. BRANCH AND REGIONAL PERFORMANCE")

branches["Default_Rate"] = branches["Delinquent_Loans"] / branches["Total_Active_Loans"].replace(0, np.nan)
branches["Default_Rate"] = branches["Default_Rate"].clip(upper=1)

branch_rank = branches.sort_values("Loan_Disbursement_Amount", ascending=False)
print(branch_rank[["Branch_ID", "Branch_Name", "Region", "Loan_Disbursement_Amount", "Default_Rate"]].head())

plt.figure(figsize=(8, 5))
sns.barplot(data=branches, x="Region", y="Loan_Disbursement_Amount", estimator=sum)
plt.title("Loan Disbursement by Region")
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_PATH, "04_region_disbursement.png"))
plt.close()


# =============================================================================
# 5. CUSTOMER SEGMENTATION
# =============================================================================

print("\n5. CUSTOMER SEGMENTATION")

loan_data["Income_Group"] = pd.cut(
    loan_data["Annual_Income"],
    bins=[0, 300000, 700000, 1200000, 999999999],
    labels=["Low", "Medium", "High", "Very High"],
)

loan_data["Credit_Group"] = pd.cut(
    loan_data["Credit_Score"],
    bins=[0, 550, 650, 750, 900],
    labels=["Poor", "Fair", "Good", "Excellent"],
)

segment = loan_data.groupby(["Income_Group", "Credit_Group"])["Default_Flag"].mean().reset_index()
print(segment)

plt.figure(figsize=(8, 5))
sns.barplot(data=segment, x="Credit_Group", y="Default_Flag", hue="Income_Group")
plt.title("Default Rate by Income and Credit Segment")
plt.savefig(os.path.join(OUTPUT_PATH, "05_customer_segmentation.png"))
plt.close()


# =============================================================================
# 6. ADVANCED STATISTICAL ANALYSIS
# =============================================================================

print("\n6. ADVANCED STATISTICAL ANALYSIS")

advanced_columns = [
    "Credit_Score",
    "Loan_Amount",
    "Interest_Rate",
    "Overdue_Amount",
    "EMI_Amount",
    "Recovery_Rate",
    "Default_Flag",
]

advanced_corr = loan_data[advanced_columns].corr()
print(advanced_corr)

plt.figure(figsize=(9, 6))
sns.heatmap(advanced_corr, annot=True, cmap="viridis")
plt.title("Advanced Correlation Heatmap")
plt.savefig(os.path.join(OUTPUT_PATH, "06_advanced_heatmap.png"))
plt.close()


# =============================================================================
# 7. TRANSACTION AND RECOVERY ANALYSIS
# =============================================================================

print("\n7. TRANSACTION AND RECOVERY ANALYSIS")

transaction_summary = transactions.groupby("Payment_Type")[["Amount", "Overdue_Fee"]].sum()
print(transaction_summary)

recovery_by_reason = defaults.groupby("Default_Reason")[["Default_Amount", "Recovery_Amount"]].sum()
recovery_by_reason["Recovery_Rate"] = recovery_by_reason["Recovery_Amount"] / recovery_by_reason["Default_Amount"]
print(recovery_by_reason)

plt.figure(figsize=(8, 5))
recovery_by_reason["Recovery_Rate"].plot(kind="bar")
plt.title("Recovery Rate by Default Reason")
plt.ylabel("Recovery Rate")
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_PATH, "07_recovery_by_reason.png"))
plt.close()


# =============================================================================
# 8. EMI ANALYSIS
# =============================================================================

print("\n8. EMI ANALYSIS")

loan_data["EMI_Band"] = pd.qcut(loan_data["EMI_Amount"], q=5, duplicates="drop")
emi_analysis = loan_data.groupby("EMI_Band")["Default_Flag"].mean()
print(emi_analysis)

plt.figure(figsize=(8, 5))
emi_analysis.plot(kind="bar")
plt.title("Default Rate by EMI Band")
plt.ylabel("Default Rate")
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_PATH, "08_emi_analysis.png"))
plt.close()


# =============================================================================
# 9. LOAN APPLICATION INSIGHTS
# =============================================================================

print("\n9. LOAN APPLICATION INSIGHTS")

approval_counts = applications["Approval_Status"].value_counts()
print(approval_counts)
print("Approval Rate:", (applications["Approval_Status"] == "Approved").mean())

rejection_reasons = applications["Rejection_Reason"].value_counts().head(10)
print(rejection_reasons)

plt.figure(figsize=(8, 5))
approval_counts.plot(kind="bar")
plt.title("Application Approval Status")
plt.savefig(os.path.join(OUTPUT_PATH, "09_approval_status.png"))
plt.close()


# =============================================================================
# 10. RECOVERY EFFECTIVENESS
# =============================================================================

print("\n10. RECOVERY EFFECTIVENESS")

defaults["Recovery_Rate"] = defaults["Recovery_Amount"] / defaults["Default_Amount"].replace(0, np.nan)
legal_recovery = defaults.groupby("Legal_Action")["Recovery_Rate"].mean()
print(legal_recovery)

plt.figure(figsize=(7, 5))
legal_recovery.plot(kind="bar")
plt.title("Recovery Rate With and Without Legal Action")
plt.ylabel("Average Recovery Rate")
plt.savefig(os.path.join(OUTPUT_PATH, "10_legal_recovery.png"))
plt.close()


# =============================================================================
# 11. LOAN DISBURSEMENT EFFICIENCY
# =============================================================================

print("\n11. LOAN DISBURSEMENT EFFICIENCY")

print("Average processing days:", loan_data["Processing_Days"].mean())

purpose_processing = loan_data.groupby("Loan_Purpose")["Processing_Days"].mean().sort_values()
print(purpose_processing)

plt.figure(figsize=(8, 5))
purpose_processing.plot(kind="bar")
plt.title("Average Processing Days by Loan Purpose")
plt.ylabel("Days")
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_PATH, "11_processing_days.png"))
plt.close()


# =============================================================================
# 12. PROFITABILITY ANALYSIS
# =============================================================================

print("\n12. PROFITABILITY ANALYSIS")

total_interest_income = loan_data["Estimated_Interest_Income"].sum()
print("Total estimated interest income:", total_interest_income)

profit_by_purpose = loan_data.groupby("Loan_Purpose")["Estimated_Interest_Income"].sum().sort_values(ascending=False)
print(profit_by_purpose)

plt.figure(figsize=(8, 5))
profit_by_purpose.plot(kind="bar")
plt.title("Estimated Interest Income by Loan Purpose")
plt.ylabel("Interest Income")
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_PATH, "12_profit_by_purpose.png"))
plt.close()


# =============================================================================
# 13. GEOSPATIAL ANALYSIS
# =============================================================================

print("\n13. GEOSPATIAL ANALYSIS")

region_loans = loan_data.groupby("Region")["Loan_ID"].count().sort_values(ascending=False)
region_defaults = loan_data.groupby("Region")["Default_Flag"].mean().sort_values(ascending=False)

print("Active loan distribution by region:")
print(region_loans)
print("Default rate by region:")
print(region_defaults)

plt.figure(figsize=(8, 5))
region_defaults.plot(kind="bar")
plt.title("Default Rate by Region")
plt.ylabel("Default Rate")
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_PATH, "13_region_default_rate.png"))
plt.close()


# =============================================================================
# 14. DEFAULT TRENDS
# =============================================================================

print("\n14. DEFAULT TRENDS")

defaults["Default_Month"] = defaults["Default_Date"].dt.to_period("M").astype(str)
monthly_defaults = defaults.groupby("Default_Month")["Default_ID"].count()
print(monthly_defaults.head())

plt.figure(figsize=(10, 5))
monthly_defaults.plot()
plt.title("Monthly Default Trend")
plt.ylabel("Number of Defaults")
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_PATH, "14_monthly_defaults.png"))
plt.close()


# =============================================================================
# 15. BRANCH EFFICIENCY
# =============================================================================

print("\n15. BRANCH EFFICIENCY")

efficient_branches = branches.sort_values("Avg_Processing_Time")
print(efficient_branches[["Branch_ID", "Branch_Name", "Avg_Processing_Time", "Default_Rate"]].head())

plt.figure(figsize=(8, 5))
sns.scatterplot(data=branches, x="Avg_Processing_Time", y="Default_Rate", hue="Region")
plt.title("Branch Processing Time vs Default Rate")
plt.savefig(os.path.join(OUTPUT_PATH, "15_branch_efficiency.png"))
plt.close()


# =============================================================================
# 16. TIME-SERIES ANALYSIS
# =============================================================================

print("\n16. TIME-SERIES ANALYSIS")

loan_data["Disbursal_Month"] = loan_data["Disbursal_Date"].dt.to_period("M").astype(str)
monthly_disbursement = loan_data.groupby("Disbursal_Month")["Loan_Amount"].sum()
print(monthly_disbursement.head())

plt.figure(figsize=(10, 5))
monthly_disbursement.plot()
plt.title("Monthly Loan Disbursement Trend")
plt.ylabel("Loan Amount")
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_PATH, "16_monthly_disbursement.png"))
plt.close()


# =============================================================================
# 17. CUSTOMER BEHAVIOR ANALYSIS
# =============================================================================

print("\n17. CUSTOMER BEHAVIOR ANALYSIS")

loan_data["Repayment_Behavior"] = "Good"
loan_data.loc[loan_data["Overdue_Amount"] > 0, "Repayment_Behavior"] = "Occasional Delay"
loan_data.loc[loan_data["Default_Flag"] == 1, "Repayment_Behavior"] = "Defaulter"

behavior_count = loan_data["Repayment_Behavior"].value_counts()
print(behavior_count)

plt.figure(figsize=(8, 5))
behavior_count.plot(kind="bar")
plt.title("Customer Repayment Behavior")
plt.savefig(os.path.join(OUTPUT_PATH, "17_repayment_behavior.png"))
plt.close()


# =============================================================================
# 18. RISK ASSESSMENT
# =============================================================================

print("\n18. RISK ASSESSMENT")

loan_data["Risk_Level"] = "Low Risk"
loan_data.loc[(loan_data["Credit_Score"] < 650) | (loan_data["EMI_to_Income"] > 0.4), "Risk_Level"] = "Medium Risk"
loan_data.loc[(loan_data["Credit_Score"] < 550) | (loan_data["Default_Flag"] == 1), "Risk_Level"] = "High Risk"

risk_matrix = loan_data.groupby(["Loan_Purpose", "Risk_Level"])["Loan_ID"].count()
print(risk_matrix)

plt.figure(figsize=(8, 5))
loan_data["Risk_Level"].value_counts().plot(kind="bar")
plt.title("Risk Level Distribution")
plt.savefig(os.path.join(OUTPUT_PATH, "18_risk_levels.png"))
plt.close()


# =============================================================================
# 19. TIME TO DEFAULT ANALYSIS
# =============================================================================

print("\n19. TIME TO DEFAULT ANALYSIS")

time_to_default = loan_data[loan_data["Default_Flag"] == 1]["Time_to_Default_Days"]
print("Average time to default in days:", time_to_default.mean())

purpose_time_default = loan_data[loan_data["Default_Flag"] == 1].groupby("Loan_Purpose")["Time_to_Default_Days"].mean()
print(purpose_time_default)

plt.figure(figsize=(8, 5))
purpose_time_default.plot(kind="bar")
plt.title("Average Time to Default by Loan Purpose")
plt.ylabel("Days")
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_PATH, "19_time_to_default.png"))
plt.close()


# =============================================================================
# 20. TRANSACTION PATTERN ANALYSIS
# =============================================================================

print("\n20. TRANSACTION PATTERN ANALYSIS")

payment_type_count = transactions["Payment_Type"].value_counts()
print(payment_type_count)

penalty_share = transactions["Overdue_Fee"].sum() / transactions["Amount"].sum()
print("Penalty as proportion of total transaction amount:", penalty_share)

plt.figure(figsize=(8, 5))
payment_type_count.plot(kind="bar")
plt.title("Transaction Count by Payment Type")
plt.savefig(os.path.join(OUTPUT_PATH, "20_transaction_patterns.png"))
plt.close()


# =============================================================================
# FINAL SUMMARY AND CONCLUSION
# =============================================================================

print("\nFINAL SUMMARY")
print("1. The project cleaned and merged customer, loan, application, transaction, default, and branch data.")
print("2. Default risk was studied using credit score, loan amount, EMI, overdue amount, and income.")
print("3. Branch and regional performance were compared using disbursement, processing time, and default rate.")
print("4. Customer segments were created using income group, credit group, repayment behavior, and risk level.")
print("5. Profitability was estimated using loan amount, interest rate, and loan term.")
print("6. Recovery effectiveness was checked using recovery amount divided by default amount.")

print("\nCONCLUSION")
print("Hero FinCorp can reduce defaults by focusing on low credit score customers, high EMI burden customers,")
print("high overdue loans, risky loan purposes, and branches with high default rates. The company should improve")
print("approval checks, monitor repayment behavior early, strengthen recovery actions, and support high-value")
print("customers with better service.")

print("\nAll 20 tasks completed. Charts saved in:", OUTPUT_PATH)

