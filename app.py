import streamlit as st # Εισάγει το Streamlit για δημιουργία διαδραστικής web εφαρμογής
import pandas as pd # Εισάγει την pandas για ανάγνωση και επεξεργασία αρχείων CSV
import joblib # Εισάγει τη joblib για φόρτωση του εκπαιδευμένου μοντέλου
import numpy as np # Εισάγει τη numpy για αριθμητικούς πίνακες (arrays)
from rdkit import Chem # Εισάγει το RDKit Chem module για διαχείριση μορίων
from rdkit.Chem import Draw # Εισάγει εργαλείο του RDKit για σχεδίαση μοριακών εικόνων
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator # Εισάγει generator για Morgan fingerprints
from rdkit import DataStructs # Εισάγει το RDKit module για μετατροπή fingerprints σε numpy arrays
import matplotlib.pyplot as plt# Εισάγει τη matplotlib για παραγωγή γραφημάτων


# -----------------------------------------------------
# Ρυθμίσεις εφαρμογής
# -----------------------------------------------------
st.set_page_config(page_title="Drug Activity Predictor", layout="wide") # Ορίζει τίτλο καρτέλας browser και κάνει τη σελίδα να καταλαμβάνει όλο το πλάτος της οθόνης
# Custom CSS
st.markdown("""
<style>
body {
    background-color: #f7f9fb;
}
.main {
    background-color: #ffffff;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)
# Ενσωματώνει CSS για να αλλάξει το φόντο και τη μορφή της εφαρμογής (custom εμφάνιση)

# -----------------------------------------------------
# Φόρτωση λογότυπου Πανεπιστημίου
# -----------------------------------------------------
st.sidebar.image("images.jpg", width=150) # Εμφανίζει το λογότυπο του πανεπιστημίου στο sidebar
st.sidebar.markdown("**Ιόνιο Πανεπιστήμιο — Τμήμα Πληροφορικής**") # Εμφανίζει bold κείμενο στο sidebar
st.sidebar.markdown("---") # Εμφανίζει οριζόντια διαχωριστική γραμμή

# -----------------------------------------------------
# Επικεφαλίδα κύριας σελίδας
# -----------------------------------------------------
st.title("Drug Activity Predictor") # Εμφανίζει τον κύριο τίτλο της εφαρμογής
st.markdown("""
### Πτυχιακή Εργασία — Δομική Βιοπληροφορική & Υπολογιστικός Σχεδιασμός Φαρμάκων  
Προέβλεψε την πιθανή δραστικότητα μορίων με **Μηχανική Μάθηση (Random Forest)** και **RDKit fingerprints**.
""") # Εμφανίζει υπότιτλο και σύντομη περιγραφή της εφαρμογής

# -----------------------------------------------------
# Φόρτωση μοντέλου & generator
# -----------------------------------------------------
model = joblib.load("bace1_model.joblib") # Φορτώνει το εκπαιδευμένο μοντέλο Random Forest από το αρχείο joblib
gen = GetMorganGenerator(radius=2, fpSize=1024)
# Δημιουργεί Morgan Fingerprint Generator:
# radius=2 → μέγεθος "γειτονιάς"
# fpSize=1024 → μέγεθος του binary fingerprint (1024 bits)

# -----------------------------------------------------
# Επιλογή λειτουργίας
# -----------------------------------------------------
mode = st.sidebar.radio("Επίλεξε λειτουργία:", ["🔹 Μοναδικό μόριο", "📂 CSV αρχείο"]) # Δημιουργεί radio button στο sidebar με δύο επιλογές:
# → Πρόβλεψη για ένα μόνο μόριο
# → Πρόβλεψη για πολλά μόρια μέσω CSV
st.sidebar.info("Μπορείς να αλλάξεις λειτουργία από εδώ.")
# Εμφανίζει μπλε πλαίσιο με πληροφορία
# -----------------------------------------------------
# Ιστορικό εκτελέσεων (session state)
# -----------------------------------------------------
if "history" not in st.session_state: # Ελέγχει αν το session (τρέχουσα εκτέλεση) έχει ιστορικό αποθηκευμένο
    st.session_state["history"] = pd.DataFrame(columns=["SMILES", "Prediction", "Probability"]) # Αν δεν υπάρχει ιστορικό → δημιουργεί νέο κενό DataFrame με 3 στήλες

# -----------------------------------------------------
# ✅ Περίπτωση 1: Μοναδικό μόριο
# -----------------------------------------------------
if mode == "🔹 Μοναδικό μόριο": # Ελέγχει ποια επιλογή έκανε ο χρήστης στο sidebar
    smiles_input = st.text_input("Δώσε SMILES μορίου:", "") # Προβάλλει πεδίο εισαγωγής κειμένου όπου ο χρήστης γράφει το SMILES του μορίου
    if smiles_input: # Αν υπάρχει κείμενο στο πεδίο εισαγωγής
        mol = Chem.MolFromSmiles(smiles_input) # Μετατρέπει το SMILES string σε αντικείμενο μορίου του RDKit
        if mol:
            fp = gen.GetFingerprint(mol) #Δημιουργεί το Morgan Fingerprint (1024-bit) για το μόριο
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr) # Μετατρέπει το RDKit fingerprint σε numpy array (bit vector
            prediction = model.predict([arr])[0] # Το Random Forest προβλέπει 0 (Inactive) ή 1 (Active)
            prob = model.predict_proba([arr])[0][1] # Υπολογίζει την πιθανότητα να ανήκει στην κατηγορία "Active" (1)

            col1, col2 = st.columns(2) # Διαχωρίζει τη σελίδα σε δύο κάθετες στήλες
            with col1: # Εμφανίζει την εικόνα του μορίου σε αριστερή στήλη
                st.image(Draw.MolToImage(mol, size=(300, 300)))
            with col2:
                if prediction == 1:
                    st.success(f"🧬 ΠΙΘΑΝΟΝ ΕΝΕΡΓΟ (πιθανότητα {prob:.2f})") # Αν το μοντέλο προβλέπει ενεργό → πράσινο μήνυμα επιτυχίας
                else:
                    st.error(f"💤 ΠΙΘΑΝΟΝ ΑΝΕΝΕΡΓΟ (πιθανότητα {prob:.2f})") # Αν είναι ανενεργό → κόκκινο μήνυμα

            # Αποθήκευση στο ιστορικό
            st.session_state["history"].loc[len(st.session_state["history"])] = [smiles_input,
                                                                                  "Active" if prediction == 1 else "Inactive",
                                                                                  round(prob, 2)] # Προσθέτει τη νέα πρόβλεψη ως νέα γραμμή στο ιστορικό ]

# -----------------------------------------------------
# ✅ Περίπτωση 2: CSV αρχείο με πολλά μόρια
# -----------------------------------------------------
else: # Το 'else' συνδέεται με το if mode == " Μοναδικό μόριο"
    # Άρα αυτό το κομμάτι τρέχει όταν ΔΕΝ είμαστε στη λειτουργία 'Μοναδικό μόριο'
    uploaded_file = st.file_uploader("Ανέβασε CSV αρχείο με στήλη 'smiles':", type=["csv"]) # Δημιουργεί κουμπί για ανέβασμα αρχείου .csv
    # Ο χρήστης πρέπει να δώσει αρχείο που περιέχει στήλη με όνομα 'smiles'
    if uploaded_file:
        df = pd.read_csv(uploaded_file) # Διαβάζει το ανεβασμένο CSV και το κάνει DataFrame

        if 'smiles' not in df.columns:
            st.error("⚠️ Το αρχείο πρέπει να περιέχει στήλη 'smiles'.") # Αν η στήλη δεν υπάρχει → εμφανίζει μήνυμα λάθους
        else:
            preds, probs = [], [] # Δημιουργεί δύο κενές λίστες:
            # preds → οι προβλέψεις Active/Inactive
            # probs → πιθανότητες
            for smi in df['smiles']: # Επαναλαμβάνει για κάθε μόριο στο CSV
                mol = Chem.MolFromSmiles(smi) # Μετατρέπει το SMILES σε μόριο RDKit
                if mol:
                    fp = gen.GetFingerprint(mol) # Υπολογίζει το Morgan fingerprint του μορίου
                    arr = np.zeros((1,)) # Δημιουργεί array 1024 στοιχείων για την αποθήκευση του fingerprint
                    DataStructs.ConvertToNumpyArray(fp, arr) # Μετατρέπει το RDKit fingerprint σε numpy array
                    pred = model.predict([arr])[0] # Προβλέπει 0 = inactive, 1 = active
                    prob = model.predict_proba([arr])[0][1] # Πιθανότητα για Active (κατηγορία 1)
                    preds.append("Active" if pred == 1 else "Inactive") # Μετατρέπει τον αριθμό σε κείμενο ("Active" ή "Inactive")
                    probs.append(round(prob, 2)) # Αποθηκεύει την πιθανότητα
                else:
                    preds.append("Invalid SMILES")
                    probs.append(None)
            # Για άκυρα SMILES μπαίνει ειδική ένδειξη
            df['Prediction'] = preds # Προσθέτει δύο νέες στήλες στο df με προβλέψεις & πιθανότητες
            df['Probability'] = probs

            # --- Εμφάνιση πίνακα ---
            st.subheader("📊 Αποτελέσματα Πρόβλεψης")
            st.dataframe(df)
            # Εμφανίζει τον πίνακα με τις προβλέψεις

            # --- Στατιστικά γραφήματα ---
            st.subheader("📈 Στατιστική Επισκόπηση")
            active_count = df['Prediction'].value_counts().get("Active", 0) # Γράφημα κατανομής
            inactive_count = df['Prediction'].value_counts().get("Inactive", 0) # Μετράει πόσα "Active" και πόσα "Inactive" προβλέφθηκαν
            fig, ax = plt.subplots()
            ax.pie(
                [active_count, inactive_count],
                labels=['Active', 'Inactive'],
                autopct='%1.1f%%',
                colors=['#28a745', '#dc3545']
            ) # Δημιουργεί γράφημα πίτας με ποσοστά και χρώματα
            ax.set_title("Κατανομή Πρόβλεψης Μορίων") # Ορίζει τίτλο
            st.pyplot(fig) # Το εμφανίζει στη Streamlit εφαρμογή

            # --- Κατέβασμα αποτελεσμάτων ---
            csv_download = df.to_csv(index=False).encode('utf-8') # Μετατρέπει το df σε CSV και σε byte format για λήψη
            st.download_button("💾 Κατέβασε τα αποτελέσματα (CSV)", csv_download, "predictions.csv", "text/csv") # Προσθέτει κουμπί λήψης για το αρχείο

            # --- Προσθήκη στο ιστορικό ---
            st.session_state["history"] = pd.concat([st.session_state["history"], df[["smiles", "Prediction", "Probability"]].rename(columns={"smiles": "SMILES"})], ignore_index=True)
            # Συνδέει το προηγούμενο ιστορικό με τα νέα αποτελέσματα

# -----------------------------------------------------
# 🧾 Εμφάνιση ιστορικού εκτελέσεων
# -----------------------------------------------------
st.markdown("---") # Προσθέτει μια οριζόντια διαχωριστική γραμμή για οργάνωση της σελίδας
st.subheader("Ιστορικό Εκτελέσεων") # Εμφανίζει υπότιτλο
# Το Ιστορικό καταγράφει όλες τις προηγούμενες προβλέψεις
if not st.session_state["history"].empty: # Ελέγχει αν το DataFrame ιστορικού *δεν* είναι άδειο
    st.dataframe(st.session_state["history"])
    # Εμφανίζει τον πίνακα με:
    # - SMILES
    # - Prediction (Active / Inactive / Invalid)
    # - Probability
else:
    st.info("Δεν υπάρχουν προηγούμενες εκτελέσεις.") # Εμφανίζει μήνυμα ενημέρωσης όταν το ιστορικό είναι άδειο

st.markdown("---") # Ξανά οριζόντια γραμμή
st.caption("© 2025 Ιόνιο Πανεπιστήμιο — Πτυχιακή Εργασία: Δομική Βιοπληροφορική & Υπολογιστικός Σχεδιασμός Φαρμάκων")
