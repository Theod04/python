# training_script.py

# --------------------------------------
# 📘 Εισαγωγή βιβλιοθηκών
# --------------------------------------
import pandas as pd  # Χρήση pandas γα φόρτωση κ διαχείριση δεδομένων σε μορφή CSV
import numpy as np  # Αριθμητικά arrays, απαραίτητα για τη μετατροπή fingerprints
from rdkit import Chem  # Βιβλιοθήκη RDKit για διαχείριση χημικών μορίων
from rdkit.Chem import AllChem # Χρησιμοποιείται για τον υπολογισμό Morgan fingerprints
from rdkit import DataStructs # Μετατροπή fingerprints to numpy vectors
from sklearn.ensemble import RandomForestClassifier # Αλγόριθμος Random forest για ταξινόμηση
from sklearn.model_selection import train_test_split # Διαχωρισμός δεδομένων σε train/test
from sklearn.metrics import accuracy_score, roc_auc_score # Μετρικές αξιολόγησης
import joblib # Αποθήκευση εκπαιδευμένου μοντέλου σε αρχείο
import matplotlib.pyplot as plt # Γραφήματα (ROC curve)
from sklearn.metrics import RocCurveDisplay # Αυτόματη δημιουργία ROC curve


# --------------------------------------
# 📘 1. Φόρτωση δεδομένων
# --------------------------------------
df = pd.read_csv("drug_docking_dataset.csv")
# Φορτώνει τα δεδομένα από CSV.
# Το αρχείο *πρέπει* να περιέχει 2 στήλες:
#   - 'smiles' : η χημική δομή του μορίου
#   - 'active' : 1 (ενεργό) ή 0 (ανενεργό)

# --------------------------------------
# 📘 2. Υπολογισμός Morgan Fingerprints
# --------------------------------------
fingerprints = [] # Λίστα που θα αποθηκεύσει τα fingerprints όλων των μορίων
for smi in df['smiles']: # Επανάληψη για κάθε SMILES στο dataset
    mol = Chem.MolFromSmiles(smi)
    # Μετατρέπει το SMILES string σε RDKit molecule object
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    # Υπολογισμός Morgan fingerprint (ECFP)
    # radius=2 → πόσα χημικά "βήματα" εξετάζει γύρω από κάθε άτομο
    # nBits=1024 → μέγεθος fingerprint (1024 χαρακτηριστικά)
    arr = np.zeros((1,)) # Δημιουργεί κενό numpy array που θα πάρει τα bit values
    DataStructs.ConvertToNumpyArray(fp, arr) # Μετατρέπει το RDKit fingerprint σε numpy array (άρα μπορεί να μπει σε ML αλγόριθμο)
    fingerprints.append(arr)    # Αποθηκεύει το array στη λίστα
X = np.array(fingerprints) # Μετατροπή όλων των fingerprints σε numpy πίνακα
# Το X περιέχει όλα τα χαρακτηριστικά των μορίων
y = df['active'] # Η ετικέτα (0/1) για κάθε μόριο,παίρνει τις τιμές της στήλης 'active'

# --------------------------------------
# 📘 3. Διαχωρισμός σε train/test
# --------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# test_size=0.4 → 40% των δεδομένων για test, 60% για training
# random_state=42 → παράγει πάντα τα ίδια αποτελέσματα (αναπαραγωγιμότητα)
# --------------------------------------
# 📘 4. Εκπαίδευση Random Forest
# --------------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
# n_estimators=100 → αριθμός δέντρων στο δάσος
# Random Forest είναι ισχυρός για binary classification (Active/Inactive)
model.fit(X_train, y_train)
# Εκπαιδεύει το μοντέλο πάνω στα δεδομένα training
# --------------------------------------
# 📘 5. Αξιολόγηση μοντέλου
# --------------------------------------
y_pred = model.predict(X_test) # Προβλέπει ετικέτες (0/1) για τα test δεδομένα
acc = accuracy_score(y_test, y_pred) # Υπολογισμός Accuracy = πόσα σωστά ποσοστιαία
roc = roc_auc_score(y_test, y_pred) # ROC AUC = πόσο καλά διαχωρίζει τις δύο κατηγορίες

print(f"✅ Accuracy: {acc:.2f}") # Εκτυπώνει την ακρίβεια με δύο δεκαδικά
print(f"✅ ROC AUC: {roc:.2f}") # Εκτυπώνει το ROC score

# --------------------------------------
# 📘 6. Οπτικοποίηση ROC Curve
# --------------------------------------
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve for Drug Activity Prediction")
plt.show()

# --------------------------------------
# 📘 7. Αποθήκευση μοντέλου
# --------------------------------------
joblib.dump(model, "bace1_model.joblib")
print("✅ Model saved: bace1_model.joblib")
