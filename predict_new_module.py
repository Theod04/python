import os # Εισαγωγή της βιβλιοθήκης os για λειτουργίες του λειτουργικού συστήματος
import sys  # Εισαγωγή της βιβλιοθήκης sys για χειρισμό του συστήματος
import joblib # Εισαγωγή της βιβλιοθήκης joblib για φόρτωση/αποθήκευση μοντέλων μηχανικής μάθησης
import numpy as np # Εισαγωγή της βιβλιοθήκης numpy για αριθμητικούς υπολογισμούς
from rdkit import Chem # Εισαγωγή του Chem από το rdkit για χειρισμό χημικών δομών
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator # Εισαγωγή του GetMorganGenerator για παραγωγή αποτυπωμάτων Morgan
from rdkit import DataStructs # Εισαγωγή του DataStructs για χειρισμό αποτυπωμάτων

# 🔇 Σίγαση όλων των μηνυμάτων RDKit (προαιρετικό)
sys.stderr = open(os.devnull, 'w') # Απόκλιση των μηνυμάτων λάθους στο devnull, ώστε να  μην εμφανίζονται

# ✅ Φόρτωση εκπαιδευμένου μοντέλου
model = joblib.load("bace1_model.joblib") # Φόρτωση του μοντέλου από το αρχείο

# 🧪 Εισαγωγή SMILES από τον χρήστη
new_smiles = input("Δώσε SMILES μορίου: ") # Ζήτηση εισόδου SMILES από τον χρήστη

# Μετατροπή του SMILES σε RDKit Mol
mol = Chem.MolFromSmiles(new_smiles) # Μετατροπή του SMILES σε αντικείμενο Mol
if mol is None:
    print("⚠️ Το SMILES δεν είναι έγκυρο. Δοκίμασε ξανά.") # Εμφάνιση μηνύματος αν το SMILES δεν είναι έγκυρο
    exit() # Τερματισμός προγράμματος

# ✅ Υπολογισμός Morgan Fingerprint με το νέο API
gen = GetMorganGenerator(radius=2, fpSize=1024) # Δημιουργία generator για Morgan Fingerprint
fp = gen.GetFingerprint(mol) # Υπολογισμός του fingerprint για το μόριο
arr = np.zeros((1,)) # Δημιουργία ενός μηδενικού numpy array
DataStructs.ConvertToNumpyArray(fp, arr) # Μετατροπή του fingerprint σε numpy array

# 🔮 Πρόβλεψη
prediction = model.predict([arr])[0] # Πρόβλεψη της κλάσης (ενεργό/ανενεργό)
prob = model.predict_proba([arr])[0][1] # Υπολογισμός πιθανότητας για την ενεργή κλάση

# 🧬 Εμφάνιση αποτελέσματος
if prediction == 1:
    print(f"🧬 Το μόριο ΠΙΘΑΝΟΝ είναι ΕΝΕΡΓΟ (πιθανότητα {prob:.2f})") # Εμφάνιση μηνύματος αν προβλέπεται ενεργό
else:
    print(f"💤 Το μόριο ΠΙΘΑΝΟΝ είναι ΑΝΕΝΕΡΓΟ (πιθανότητα {prob:.2f})")  # Εμφάνιση μηνύματος αν προβλέπεται ανενεργό