import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
st.title('🤖Machine Learning app')
st.info('This app builds an Recommendation model!')
# 📌 STEP 2: UPLOAD CSV FILE
from google.colab import files
uploaded = files.upload()

df = pd.read_csv(next(iter(uploaded)))

# 📌 STEP 3: USER INPUT
period = int(input("Enter number of periods to forecast ahead: "))

# 📌 STEP 4: DEFINE COLUMNS
name_col = 'name'
target_col = 'profit'

if name_col not in df.columns or target_col not in df.columns:
    raise ValueError(f"Dataset must include '{name_col}' and '{target_col}' columns.")

# 📌 STEP 5: PREPARE FEATURES
X = df.drop(columns=[target_col, name_col])
y = df[target_col]

# Save employee names for reporting
employee_names = df[name_col]

# Filter to only numeric columns
X_numeric = X.select_dtypes(include=[np.number])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# 📌 STEP 6: TRAIN MODEL
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 STEP 7: PREDICT FUTURE PROFITS
# Add growth simulation (noise) for the future period
predicted_profit = model.predict(X_scaled) + (period * np.random.normal(loc=0.2, scale=0.3, size=len(X_scaled)))

# Add prediction to DataFrame
df['Predicted Profit After Period'] = predicted_profit

# 📌 STEP 8: COMPARISON TABLE
comparison_df = df[[name_col] + list(X_numeric.columns) + ['Predicted Profit After Period']]
sorted_comparison = comparison_df.sort_values(by='Predicted Profit After Period', ascending=False)

print("\n📊 Top 10 Employees by Predicted Profitability:\n")
print(sorted_comparison.head(10).to_string(index=False))

# 📌 STEP 9: MOST PROFITABLE EMPLOYEE
top_employee = sorted_comparison.iloc[0]
print("\n🏆 Most Profitable Employee After", period, "Periods:")
print(f"Name: {top_employee[name_col]}")
print(top_employee)

# 📌 STEP 10: VISUALIZATION
plt.figure(figsize=(12, 6))
sns.barplot(x=sorted_comparison[name_col][:10], y=sorted_comparison['Predicted Profit After Period'][:10])
plt.title(f"Top 10 Employees - Predicted Profit After {period} Periods")
plt.ylabel("Predicted Profit")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
