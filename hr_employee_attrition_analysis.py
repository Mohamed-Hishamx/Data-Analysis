# -----------------------------
# 1) Imports
# -----------------------------
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 110


# -----------------------------
# 2) Download dataset using kagglehub
# -----------------------------
import kagglehub

# IBM HR dataset (Kaggle)
# https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
dataset_path = kagglehub.dataset_download(
    "pavansubhasht/ibm-hr-analytics-attrition-dataset"
)

print(f"Dataset downloaded to: {dataset_path}")

# List all files in the downloaded folder
all_files = glob.glob(f"{dataset_path}/**/*", recursive=True)
print("\nFiles available:")
for f in all_files:
    if os.path.isfile(f):
        print(" ", f)


# -----------------------------
# 3) Load the CSV
# -----------------------------
csv_files = glob.glob(f"{dataset_path}/**/*.csv", recursive=True)
if not csv_files:
    raise FileNotFoundError("No CSV file found in the dataset folder.")

# Usually it's WA_Fn-UseC_-HR-Employee-Attrition.csv
# but we’ll just pick the first CSV to be safe
file_path = csv_files[0]
print(f"\nLoading: {file_path}")

df = pd.read_csv(file_path)
print("Shape:", df.shape)
df.head()


# -----------------------------
# 4) Data Cleaning
# -----------------------------
print("\nMissing values:", int(df.isna().sum().sum()))

# Drop constant columns + EmployeeNumber (ID)
constant_cols = [c for c in df.columns if df[c].nunique() == 1]
drop_cols = constant_cols + (["EmployeeNumber"] if "EmployeeNumber" in df.columns else [])
df_clean = df.drop(columns=drop_cols).copy()

# Binary target for easier math
df_clean["Attrition_Flag"] = df_clean["Attrition"].map({"Yes": 1, "No": 0})

# Practical grouping columns
df_clean["AgeGroup"] = pd.cut(
    df_clean["Age"],
    bins=[17, 25, 30, 35, 40, 50, 61],
    labels=["18-25", "26-30", "31-35", "36-40", "41-50", "51-60"]
)

print("\nDropped columns:", drop_cols)
print("Clean shape:", df_clean.shape)


# -----------------------------
# 5) EDA - Visualization 1: Overtime vs Attrition Rate
# -----------------------------
overtime_rate = (df_clean.groupby("OverTime")["Attrition_Flag"].mean() * 100).sort_index()

ax = overtime_rate.plot(kind="bar", color=["#3498db", "#e74c3c"], edgecolor="white")
ax.set_title("Attrition Rate by Overtime", fontweight="bold")
ax.set_xlabel("OverTime")
ax.set_ylabel("Attrition Rate (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()

print(f"Overtime attrition rate: {overtime_rate.get('Yes', np.nan):.1f}%")
print(f"No overtime attrition rate: {overtime_rate.get('No', np.nan):.1f}%")


# -----------------------------
# 6) EDA - Visualization 2: Attrition by Age Group
# -----------------------------
age_rate = df_clean.groupby("AgeGroup")["Attrition_Flag"].mean() * 100

ax = age_rate.plot(kind="bar", color="#2ecc71", edgecolor="white")
ax.set_title("Attrition Rate by Age Group", fontweight="bold")
ax.set_xlabel("Age Group")
ax.set_ylabel("Attrition Rate (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()


# -----------------------------
# 7) EDA - Visualization 3: Job Satisfaction vs Attrition
# -----------------------------
sat_rate = df_clean.groupby("JobSatisfaction")["Attrition_Flag"].mean() * 100

ax = sat_rate.plot(kind="bar", color="#9b59b6", edgecolor="white")
ax.set_title("Attrition Rate by Job Satisfaction", fontweight="bold")
ax.set_xlabel("Job Satisfaction (1 = Low, 4 = High)")
ax.set_ylabel("Attrition Rate (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()


# -----------------------------
# 8) EDA - Visualization 4: Correlation Heatmap (Key Features)
# -----------------------------
key_features = [
    "Attrition_Flag", "Age", "MonthlyIncome", "JobSatisfaction",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "DistanceFromHome", "NumCompaniesWorked", "TotalWorkingYears",
    "WorkLifeBalance", "EnvironmentSatisfaction", "JobLevel"
]

corr = df_clean[key_features].corr(numeric_only=True)

plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Heatmap — Key Features", fontweight="bold")
plt.tight_layout()
plt.show()


# -----------------------------
# 9) Business Conclusion (simple + human)
# -----------------------------
overall_attrition = df_clean["Attrition_Flag"].mean() * 100
ot_yes = df_clean.loc[df_clean["OverTime"] == "Yes", "Attrition_Flag"].mean() * 100
ot_no = df_clean.loc[df_clean["OverTime"] == "No", "Attrition_Flag"].mean() * 100

print("\nBusiness Conclusion:")
print("-" * 60)
print(f"Overall attrition rate: {overall_attrition:.1f}%")
print(f"Overtime attrition: {ot_yes:.1f}% vs {ot_no:.1f}% (no overtime)")
print("Main takeaways:")
print("1) Overtime is a clear risk signal — HR should track it closely.")
print("2) Younger employees leave more — onboarding and growth plans matter.")
print("3) Low job satisfaction is linked to exits — surveys + follow-up can reduce churn.")