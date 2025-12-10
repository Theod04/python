import pandas as pd

# 1. Φόρτωση του αρχικού αρχείου
df = pd.read_csv("AChE_bioactivities.csv")

# 2. Κρατάμε μόνο SMILES και pChEMBL
df = df[['canonical_smiles', 'pchembl_value']]

# 3. Αφαίρεση NaN
df = df.dropna()

# 4. Αφαίρεση διπλότυπων εγγραφών
df = df.drop_duplicates()

# 5. Δημιουργία Active/Inactive label
# Active = 1 if pChEMBL >= 6, otherwise 0
df['active'] = df['pchembl_value'].apply(lambda x: 1 if x >= 6 else 0)

# 6. Μετονομασία στήλης SMILES για απλότητα
df = df.rename(columns={'canonical_smiles': 'smiles'})

# 7. Αποθήκευση καθαρού dataset
df.to_csv("AChE_clean_dataset.csv", index=False)

print("✅ Dataset cleaned and saved as AChE_clean_dataset.csv")
print(df.head())
