import pandas as pd

# Δημιουργούμε synthetic dataset βασισμένο στα δεδομένα του paper
data = {
    'compound': ['AZD3293', 'Solanezumab', 'Donepezil', 'Rivastigmine', 'Galantamine'],
    'target_protein': ['BACE1', 'BACE1', 'AChE', 'AChE', 'AChE'],
    'pdb_id': ['4DJY', '2QZC', '1EVE', '1GQR', '1W6R'],
    'binding_energy_kcal': [-9.1, -8.5, -10.2, -8.7, -8.9],
    'rmsd': [1.25, 1.65, 1.10, 1.45, 1.30],
    'active': [1, 1, 1, 1, 1],   # 1 = ενεργό φάρμακο
    'smiles': [
        'CC(C)CNCC(O)c1ccccc1',           # AZD3293 (ενδεικτικά)
        'CC1=CC=CC=C1C(=O)NC2=CC=CC=C2',   # Solanezumab (dummy small mol)
        'COC1=CC=CC=C1C(=O)N',             # Donepezil
        'CCN(CC)CCOC(=O)C1=CC=CC=C1',      # Rivastigmine
        'CCOC1=CC=CC=C1OCCN',              # Galantamine
    ]
}

df = pd.DataFrame(data)
df.to_csv("drug_docking_dataset.csv", index=False)
print("✅ Dataset saved: drug_docking_dataset.csv")
print(df)
