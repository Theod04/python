import pandas as pd  # Εισαγωγή pandas για ανάγνωση & διαχείριση δεδομένων
from rdkit import Chem # Εργαλεία RDKit για χειρισμό SMILES → Mol αντικειμένων
from rdkit.Chem import AllChem # Περιέχει συναρτήσεις για υπολογισμό Morgan Fingerprints
from rdkit import DataStructs # Χρησιμοποιείται για μετατροπή fingerprints σε πίνακα
import numpy as np  # Για αριθμητικούς πίνακες & μετασχηματισμούς
from sklearn.ensemble import RandomForestClassifier  # Αλγόριθμος Random Forest για ταξινόμηση
from sklearn.model_selection import train_test_split # Διαχωρισμός train/test
from sklearn.metrics import accuracy_score, roc_auc_score # Μετρικές αξιολόγησης

df = pd.read_csv("drug_docking_dataset.csv") # Φόρτωμα dataset από CSV αρχείο σε DataFrame

# Υπολογισμός Morgan fingerprints (1024 bits)

fingerprints = [] # Λίστα όπου θα αποθηκευτούν όλα τα fingerprints
for smi in df['smiles']:# Επανάληψη πάνω σε κάθε SMILES στο dataset
    mol = Chem.MolFromSmiles(smi) # Μετατροπή SMILES → μόριο RDKit
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) # Morgan Fingerprint ακτίνας 2 με 1024 bits
    arr = np.zeros((1,)) # Δημιουργία πίνακα numpy (θα γεμίσει με 0 & 1 από το fingerprint)
    DataStructs.ConvertToNumpyArray(fp, arr) # Μετατροπή fingerprint σε numpy array
    fingerprints.append(arr)  # Αποθήκευση του fingerprint στη λίστα

X = np.array(fingerprints)  # Είσοδος μοντέλου (features): fingerprints όλων των μορίων
y = df['active']  # Labels: αν το μόριο είναι ενεργό (1) ή όχι (0)

# Διαχωρισμός train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# 60% train - 40% test & σταθερό random seed για ίδια αποτελέσματα κάθε φορά

# Εκπαίδευση Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Ορισμός μοντέλου με 100 δέντρα
model.fit(X_train, y_train) # Εκπαίδευση μοντέλου στα fingerprints

# Αξιολόγηση
y_pred = model.predict(X_test) # Πρόβλεψη για το test set
acc = accuracy_score(y_test, y_pred)  # Υπολογισμός Accuracy (% σωστών προβλέψεων)
roc = roc_auc_score(y_test, y_pred) # Υπολογισμός ROC-AUC (πόσο καλά διαχωρίζει κατηγορίες)

print(f"✅ Accuracy: {acc:.2f}")  # Εκτύπωση αποτελέσματος accuracy
print(f"✅ ROC AUC: {roc:.2f}")  # Εκτύπωση ROC-AUC score