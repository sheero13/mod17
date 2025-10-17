import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

df = pd.read_csv(r"Hours and Scores.csv")

print("Dataset Loaded Successfully!\n")
print(df.head())

X = df['Hours']
y = df['Scores']

r, p_value = pearsonr(X, y)
print(f"\nCorrelation Coefficient (r): {r:.3f}")
print(f"P-value: {p_value:.5f}")

alpha = 0.05
if p_value < alpha:
    print("Reject H₀: Significant correlation between study hours and exam scores.")
else:
    print("Fail to reject H₀: No significant correlation found.")

slope, intercept, r_value, p_val, std_err = linregress(X, y)
print(f"\nRegression Equation: Exam_Score = {intercept:.2f} + {slope:.2f} * Study_Hours")
print(f"R-squared: {r_value**2:.3f}")
print(f"P-value (Regression): {p_val:.5f}")

plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, intercept + slope * X, color='red', label='Regression Line')
plt.title('Study Hours vs Exam Scores')
plt.xlabel('Study Hours')
plt.ylabel('Exam Scores')
plt.legend()
plt.show()

#Interpretation
# Correlation:
# r = 0.997 → Very strong positive correlation.
# p < 0.05 → Statistically significant relationship.

# Regression:
# Regression equation:
# Exam Score = 35 + 5 × Study Hours
# R² = 0.994 → 99.4% of the variation in exam scores is explained by study hours.
# p < 0.05 → The slope is significant.

# Conclusion:
# Reject the Null Hypothesis (H₀).
# There is a strong, significant positive relationship between study hours and exam scores.
# As study time increases, exam performance tends to increase.
