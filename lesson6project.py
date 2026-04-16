
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

df = pd.read_csv("penguins.csv")


print("First 5 rows:\n", df.head())

print("\nNull values count:\n", df.isnull().sum())

plt.figure(figsize=(8, 5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns


for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)


for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nNull values after cleaning:\n", df.isnull().sum())

sns.pairplot(df, hue="species")  # adjust column if needed
plt.suptitle("Pairplot of Numerical Features", y=1.02)
plt.show()

plt.figure(figsize=(10, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


gender_col = "sex" if "sex" in df.columns else "gender"

plt.figure(figsize=(6, 4))
sns.countplot(x=df[gender_col], palette="Set2")
plt.title("Countplot of Gender")
plt.show()

print("\nAnalysis Completed Successfully!")