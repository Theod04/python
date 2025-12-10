import pandas as pd
# Εισάγει τη βιβλιοθήκη pandas, η οποία χρησιμοποιείται για διαχείριση δεδομένων
# και για τη δημιουργία/αποθήκευση του CSV αρχείου.

# ΝΕΟ dataset με ενεργά (1) και ανενεργά (0) μόρια
data = { # Ορίζουμε ένα dictionary όπου κάθε κλειδί είναι στήλη του dataset
    # και κάθε τιμή είναι λίστα με τα στοιχεία της στήλης.
    'compound': [
        'AZD3293', 'Solanezumab', 'Donepezil', 'Rivastigmine', 'Galantamine',
        'Dummy1', 'Dummy2', 'Dummy3'
    ],# Όνομα φαρμακευτικού μορίου ή ένωσης (π.χ. πραγματικά + ψεύτικα δεδομένα για training)
    'target_protein': ['BACE1', 'BACE1', 'AChE', 'AChE', 'AChE', 'BACE1', 'AChE', 'BACE1'],     # Πρωτεΐνη-στόχος που αλληλεπιδρά το μόριο (π.χ. BACE1, AChE)
    'pdb_id': ['4DJY', '2QZC', '1EVE', '1GQR', '1W6R', '0XYZ', '0XYZ', '0XYZ'],    # PDB ID της πρωτεΐνης σε docking / μοντελοποίηση
    'binding_energy_kcal': [-9.1, -8.5, -10.2, -8.7, -8.9, -5.2, -6.1, -4.8],  # Ενεργεια δέσμευσης docking (kcal/mol) → χαμηλότερη τιμή = καλύτερη δέσμευση
    'rmsd': [1.25, 1.65, 1.10, 1.45, 1.30, 2.5, 2.7, 3.0],     # RMSD από docking → πόσο καλή είναι η ευθυγράμμιση του ligand
    'active': [1, 1, 1, 1, 1, 0, 0, 0],  # Τώρα έχουμε και 0 και 1
# Η μεταβλητή στόχος του Machine Learning (Label)
    # 1 = δραστικό μόριο, 0 = ανενεργό μόριο
    'smiles': [
        'CC(C)CNCC(O)c1ccccc1',            # AZD3293
        'CC1=CC=CC=C1C(=O)NC2=CC=CC=C2',   # Solanezumab
        'COC1=CC=CC=C1C(=O)N',             # Donepezil
        'CCN(CC)CCOC(=O)C1=CC=CC=C1',      # Rivastigmine
        'CCOC1=CC=CC=C1OCCN',              # Galantamine
        'CCCCC(=O)N',                      # Dummy inactive
        'CCCNC(=O)C1=CC=CC=C1',            # Dummy inactive
        'CCCCCCO'                           # Dummy inactive
    ]
    # Οι χημικές δομές σε μορφή SMILES που θα μετατραπούν σε fingerprints από το RDKit
}

df = pd.DataFrame(data) # Μετατρέπει το dictionary "data" σε pandas DataFrame (πίνακα δεδομένων)
df.to_csv("drug_docking_dataset.csv", index=False)
# Αποθηκεύει τον DataFrame σε CSV αρχείο με όνομα "drug_docking_dataset.csv"
# index=False → δεν αποθηκεύει τον αριθμό γραμμής στο αρχείο (καθαρότερο CSV)
print("✅ Νέο dataset αποθηκεύτηκε: drug_docking_dataset.csv") # Εκτυπώνει μήνυμα επιβεβαίωσης ότι το νέο dataset δημιουργήθηκε επιτυχώς
