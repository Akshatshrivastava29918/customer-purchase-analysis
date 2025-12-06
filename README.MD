# customer-purchase-analysis
this  customer purchase analysis in the hypothesis testing in the EDA method

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============= CREATE SAMPLE DATASET =============
np.random.seed(42)
n_samples = 500

data = {
    'customer_id': range(1, n_samples + 1),
    'age': np.random.randint(18, 75, n_samples),
    'gender': np.random.choice(['M', 'F'], n_samples),
    'purchase_amount': np.random.exponential(150, n_samples) + 20,
    'purchase_frequency': np.random.randint(1, 20, n_samples),
    'customer_segment': np.random.choice(['Premium', 'Standard', 'Budget'], n_samples),
    'days_since_last_purchase': np.random.randint(1, 365, n_samples),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home'], n_samples),
}

df = pd.DataFrame(data)
df['total_spending'] = df['purchase_amount'] * df['purchase_frequency']

print("="*70)
print("CUSTOMER PURCHASE ANALYSIS - HYPOTHESIS TESTING & EDA")
print("="*70)

# ============= EXPLORATORY DATA ANALYSIS =============
print("\n--- DATASET OVERVIEW ---")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few records:\n{df.head()}")
print(f"\nBasic Statistics:\n{df.describe()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# ============= UNIVARIATE ANALYSIS =============
print("\n" + "="*70)
print("UNIVARIATE ANALYSIS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution of purchase amount
axes[0, 0].hist(df['purchase_amount'], bins=40, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Purchase Amount', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Amount ($)')

# Distribution of purchase frequency
axes[0, 1].hist(df['purchase_frequency'], bins=20, color='lightcoral', edgecolor='black')
axes[0, 1].set_title('Distribution of Purchase Frequency', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Frequency')

# Distribution of age
axes[1, 0].hist(df['age'], bins=20, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('Distribution of Customer Age', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Age')

# Customer segment distribution
segment_counts = df['customer_segment'].value_counts()
axes[1, 1].bar(segment_counts.index, segment_counts.values, color=['gold', 'silver', 'chocolate'])
axes[1, 1].set_title('Customer Segment Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('univariate_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

# ============= BIVARIATE ANALYSIS =============
print("\n" + "="*70)
print("BIVARIATE ANALYSIS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age vs Purchase Amount
axes[0, 0].scatter(df['age'], df['purchase_amount'], alpha=0.5, color='blue')
axes[0, 0].set_title('Age vs Purchase Amount', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Purchase Amount ($)')

# Purchase Frequency vs Total Spending
axes[0, 1].scatter(df['purchase_frequency'], df['total_spending'], alpha=0.5, color='green')
axes[0, 1].set_title('Purchase Frequency vs Total Spending', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Purchase Frequency')
axes[0, 1].set_ylabel('Total Spending ($)')

# Gender vs Purchase Amount (Box plot)
df.boxplot(column='purchase_amount', by='gender', ax=axes[1, 0])
axes[1, 0].set_title('Purchase Amount by Gender', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Gender')
axes[1, 0].set_ylabel('Purchase Amount ($)')

# Segment vs Total Spending (Box plot)
df.boxplot(column='total_spending', by='customer_segment', ax=axes[1, 1])
axes[1, 1].set_title('Total Spending by Customer Segment', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Customer Segment')
axes[1, 1].set_ylabel('Total Spending ($)')

plt.tight_layout()
plt.savefig('bivariate_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

# ============= CORRELATION ANALYSIS =============
print("\nCorrelation Matrix:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=100, bbox_inches='tight')
plt.show()

# ============= HYPOTHESIS TESTING =============
print("\n" + "="*70)
print("HYPOTHESIS TESTING")
print("="*70)

# H1: Mean purchase amount differs by gender
print("\n--- H1: Purchase Amount Differs by Gender ---")
male_purchases = df[df['gender'] == 'M']['purchase_amount']
female_purchases = df[df['gender'] == 'F']['purchase_amount']

t_stat, p_value = ttest_ind(male_purchases, female_purchases)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Result: {'REJECT' if p_value < 0.05 else 'FAIL TO REJECT'} null hypothesis (α=0.05)")
print(f"Interpretation: Purchase amounts {'significantly differ' if p_value < 0.05 else 'do not significantly differ'} by gender")

# H2: Customer segments have different average spending
print("\n--- H2: Total Spending Differs by Customer Segment ---")
premium = df[df['customer_segment'] == 'Premium']['total_spending']
standard = df[df['customer_segment'] == 'Standard']['total_spending']
budget = df[df['customer_segment'] == 'Budget']['total_spending']

f_stat, p_value = f_oneway(premium, standard, budget)
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Result: {'REJECT' if p_value < 0.05 else 'FAIL TO REJECT'} null hypothesis (α=0.05)")
print(f"Interpretation: Spending {'significantly differs' if p_value < 0.05 else 'does not significantly differ'} across segments")

# H3: Correlation between age and purchase frequency
print("\n--- H3: Correlation between Age and Purchase Frequency ---")
corr_coef, p_value = stats.pearsonr(df['age'], df['purchase_frequency'])
print(f"Pearson correlation: {corr_coef:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Result: {'REJECT' if p_value < 0.05 else 'FAIL TO REJECT'} null hypothesis (α=0.05)")
print(f"Interpretation: Age and purchase frequency are {'significantly correlated' if p_value < 0.05 else 'not significantly correlated'}")

# H4: Days since last purchase and purchase amount relationship
print("\n--- H4: Does Days Since Last Purchase Affect Purchase Amount? ---")
corr_coef, p_value = stats.spearmanr(df['days_since_last_purchase'], df['purchase_amount'])
print(f"Spearman correlation: {corr_coef:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Result: {'REJECT' if p_value < 0.05 else 'FAIL TO REJECT'} null hypothesis (α=0.05)")

# H5: Chi-square test - Gender and Product Category independence
print("\n--- H5: Gender and Product Category Independence ---")
contingency_table = pd.crosstab(df['gender'], df['product_category'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Result: {'REJECT' if p_value < 0.05 else 'FAIL TO REJECT'} null hypothesis (α=0.05)")
print(f"Interpretation: Gender and product category are {'dependent' if p_value < 0.05 else 'independent'}")

# ============= SUMMARY STATISTICS BY SEGMENT =============
print("\n" + "="*70)
print("SUMMARY STATISTICS BY CUSTOMER SEGMENT")
print("="*70)
segment_summary = df.groupby('customer_segment').agg({
    'purchase_amount': ['mean', 'median', 'std'],
    'purchase_frequency': ['mean', 'median'],
    'total_spending': ['mean', 'median', 'max'],
    'age': ['mean', 'std']
}).round(2)
print(segment_summary)

print("\n" + "="*70)
print("ANALYSIS COMPLETE - Visualizations saved as PNG files")
print("="*70)
