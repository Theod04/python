import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs
from rdkit.Chem import Draw
import matplotlib.pyplot as plt


# Ρυθμίσεις
# ---------------------------------------------------------
st.set_page_config(page_title="Multi-Target Predictor", layout="wide")

st.sidebar.image("images.jpg", width=150)
st.sidebar.markdown("**Εφαρμογή python εργαλείου για συλλογή μοριακής πρόσδεσης**")
st.sidebar.markdown("---")

# ---------------------------------------------------------
# Αρχεία μοντέλων
# ---------------------------------------------------------
MODEL_FILES = {
    "AChE": "AChE_model.joblib",
    "BACE1": "BACE1_model.joblib",
    "ADAM10": "ADAM10_model.joblib",
    "LRP1": "LRP1_model.joblib",
    "NMDA": "NMDA_model.joblib",
    "SIRT3": "SIRT3_model.joblib",
    "MAO-A": "MAO-A_model.joblib",
    "MAO-B": "MAO-B_model.joblib",
    "TREM2": "TREM2_model.joblib",
    "MARK2": "MARK2_model.joblib",
    "GLP1R": "GLP1R_model.joblib",
    "GSK3B": "GSK3B_model.joblib",
    "ABCA1": "ABCA1_model.joblib",
    "CSF1R": "CSF1R_model.joblib"
}

st.title("Δομική Βιοπληροφορική — Drug Activity Prediction")
st.markdown("Επίλεξε πρωτεΐνη και προέβλεψε τη δραστικότητα μορίου.")

# ---------------------------------------------------------
# Επιλογή πρωτεΐνης
# ---------------------------------------------------------
target = st.sidebar.selectbox("Επίλεξε πρωτεΐνη-στόχο:", list(MODEL_FILES.keys()))

# ---------------------------------------------------------
# Cached loader
# ---------------------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def get_generator():
    return GetMorganGenerator(radius=2, fpSize=1024)

model = load_model(MODEL_FILES[target])
gen = get_generator()

# ---------------------------------------------------------
# Επιλογή λειτουργίας
# ---------------------------------------------------------
mode = st.sidebar.radio("Λειτουργία:", ["Μοναδικό μόριο", "CSV αρχείο"])

plot_placeholder = st.empty()
# ---------------------------------------------------------
# Placeholder για γραφήματα (παντού ίδιο)
# ---------------------------------------------------------
plot_placeholder = st.empty()

# ---------------------------------------------------------
# Session history
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame(
        columns=["Target", "SMILES", "Prediction", "Probability"]
    )

# ---------------------------------------------------------
# Μοναδικό μόριο (ΚΑΘΑΡΙΣΜΟΣ ΓΡΑΦΗΜΑΤΟΣ)
# ---------------------------------------------------------
if mode == "Μοναδικό μόριο":

    # Σβήνει το γράφημα που είχε παραμείνει από CSV mode
    plot_placeholder.empty()

    smiles_input = st.text_input("Δώσε SMILES:")
    predict_button = st.button(" Predict Activity")

    if predict_button:
        if not smiles_input:
            st.error("Δώσε SMILES πρώτα.")
        else:
            mol = Chem.MolFromSmiles(smiles_input)

            if mol is None:
                st.error("Μη έγκυρο SMILES.")

            else:
                # Fingerprint
                fp = gen.GetFingerprint(mol)
                arr = np.zeros((1024,), dtype=int)
                DataStructs.ConvertToNumpyArray(fp, arr)


                # Prediction
                pred = model.predict([arr])[0]
                prob = model.predict_proba([arr])[0][1]


                # Layout για εικόνα και αποτέλεσμα
                col1, col2 = st.columns(2)

                with col1:
                    st.image(Draw.MolToImage(mol, size=(300, 300)))

                with col2:
                    if pred == 1:
                        st.success(f"ΕΝΕΡΓΟ ({prob:.2f})")
                    else:
                        st.error(f"ΑΝΕΝΕΡΓΟ ({prob:.2f})")

                # Ιστορικό
                st.session_state["history"].loc[len(st.session_state["history"])] = [
                    target,
                    smiles_input,
                    "Active" if pred == 1 else "Inactive",
                    round(prob, 2),
                ]

# ---------------------------------------------------------
# CSV mode (ΜΟΝΟ εδώ εμφανίζεται γράφημα)
# ---------------------------------------------------------
else:

    uploaded = st.file_uploader("Φόρτωσε CSV με στήλη 'smiles':", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        if "smiles" not in df.columns:
            st.error("Το αρχείο πρέπει να περιέχει στήλη 'smiles'.")
        else:
            preds, probs = [], []

            for smi in df["smiles"]:
                mol = Chem.MolFromSmiles(smi)

                if mol:
                    fp = gen.GetFingerprint(mol)
                    arr = np.zeros((1024,), dtype=int)
                    DataStructs.ConvertToNumpyArray(fp, arr)

                    p = model.predict([arr])[0]
                    pr = model.predict_proba([arr])[0][1]

                    preds.append("Active" if p == 1 else "Inactive")
                    probs.append(round(pr, 2))
                else:
                    preds.append("Invalid")
                    probs.append(None)

            df["Target"] = target
            df["Prediction"] = preds
            df["Probability"] = probs

            st.subheader("Αποτελέσματα")
            st.dataframe(df)

            # Δημιουργία Πίτας
            active = preds.count("Active")
            inactive = preds.count("Inactive")

            if active + inactive > 0:
                fig, ax = plt.subplots()
                ax.pie([active, inactive], labels=["Active", "Inactive"], autopct="%1.1f%%")

                #  ΜΟΝΟ στο CSV mode εμφανίζουμε γράφημα
                plot_placeholder.pyplot(fig)

            # Update history
            st.session_state["history"] = pd.concat(
                [st.session_state["history"],
                 df[["Target", "smiles", "Prediction", "Probability"]].rename(columns={"smiles": "SMILES"})],
                ignore_index=True
            )

# ---------------------------------------------------------
# Ιστορικό
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Ιστορικό προβλέψεων")
st.dataframe(st.session_state["history"])
