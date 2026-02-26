import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pingouin import cronbach_alpha

## PART 1: Formatting and Prep

## Get columns with likert scores
xp = pd.read_csv('/Users/djovanvelasco/Downloads/141_final_cleaned.csv')
print(xp.head())
likert_cols = xp.columns[19:34]
xp_likert = xp[likert_cols]
print(xp_likert.head())

## Impute Blanks
imputer = SimpleImputer(strategy='median')
xp_likert_imputed = pd.DataFrame(imputer.fit_transform(xp_likert), columns=xp_likert.columns)

## Reverse Coding
negative_items = [
    'hesitation_to_participate',
    'input_not_considered',
    'non_valuable_contribution',
    'feel_ignored',
    'lack_of_interaction_with_partners'
]

# Reverse score the negative items
for col in negative_items:
    xp_likert_imputed[col] = 6 - xp_likert_imputed[col]

print(xp_likert_imputed)

# PART 2: Reliability Estimate (Cronbach's Alpha) 
alpha, ci = cronbach_alpha(data=xp_likert_imputed)
print(f"Overall Cronbach's Alpha: {alpha:.3f}")
## Overall 0.762

# PART 3: Reliability Estimate with Deletion
results = []
total_scores = xp_likert_imputed.sum(axis=1)

for col in xp_likert_imputed.columns:
    # Drop the item
    df_deleted = xp_likert_imputed.drop(columns=[col])
    
    # Calculate new alpha using pingouin
    alpha_del, _ = cronbach_alpha(data=df_deleted)
    
    # Calculate item-total correlation
    item_corr = xp_likert_imputed[col].corr(total_scores)
    
    results.append({
        'Item': col,
        'Item-Total Correlation': item_corr,
        'Alpha if Deleted': alpha_del
    })

results_df = pd.DataFrame(results).round(3)
print(results_df)
results_df.to_csv('Reliability_Dropped.csv', index=False)

## PART 4: Scree Plot
scaler = StandardScaler()
xp_scaled = scaler.fit_transform(xp_likert_imputed)

# Fit Principal Component Analysis (PCA)
pca = PCA()
pca.fit(xp_scaled)
eigenvalues = pca.explained_variance_

# Plot the Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--')
plt.axhline(y=1, color='r', linestyle='-', label='Kaiser Criterion (Eigenvalue = 1)')
plt.title('Scree Plot of Likert Scale Items (Imputed & Reverse Scored)')
plt.xlabel('Principal Component (Factor Number)')
plt.ylabel('Eigenvalue')
plt.xticks(range(1, len(eigenvalues) + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save
plt.savefig('scree_plot_xp.png')
