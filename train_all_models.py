import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

TARGETS = {
    "AChE": "AChE.csv",
    "BACE1": "BACE1.csv",
    "ADAM10": "ADAM10.csv",
    "LRP1": "LRP1.csv",
    "NMDA": "NMDA.csv",
    "SIRT3": "SIRT3.csv",
    "MAO-A": "MAO-A.csv",
    "MAO-B": "MAO-B.csv",
    "TREM2": "TREM2.csv",
    "MARK2": "MARK2.csv",
    "GLP1R": "GLP1R.csv",
    "GSK3B": "GSK3B.csv",
    "ABCA1": "ABCA1.csv",
    "CSF1R": "CSF1R.csv"
}

FP_SIZE = 1024
gen = GetMorganGenerator(radius=2, fpSize=FP_SIZE)

def smiles_to_fp_array(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = gen.GetFingerprint(mol)
            arr = np.zeros((FP_SIZE,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        else:
            fps.append(np.zeros((FP_SIZE,)))
    return np.array(fps)


for target, csv_name in TARGETS.items():
    print("\n" + "="*60)
    print(f"🔹 Training model for target: {target}")
    print(f"   CSV file: {csv_name}")

    # 1. Φόρτωση dataset
    df = pd.read_csv(csv_name)

    # Αν δεν υπάρχει ήδη στήλη label, τη δημιουργούμε από το activity
    if "label" not in df.columns:
        if "activity" not in df.columns:
            raise ValueError(f"{csv_name} δεν έχει στήλη 'activity' ούτε 'label'")
        df["label"] = df["activity"].map({"Active": 1, "Inactive": 0})

    # Πετάμε γραμμές χωρίς smiles ή label
    df = df.dropna(subset=["smiles", "label"])

    print("  Δείγματα:", len(df))

    # 2. Fingerprints
    X = smiles_to_fp_array(df["smiles"])
    y = df["label"].astype(int)

    print("  Fingerprint matrix shape:", X.shape)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=42, stratify=y
    )

    # 4. Random Forest
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc = float("nan")

    print(f"  ✅ Accuracy: {acc:.3f}")
    print(f"  ✅ ROC AUC: {roc:.3f}")

    # 6. Save model με όνομα που περιμένει το Streamlit
    model_filename = f"{target}_model.joblib"
    joblib.dump(model, model_filename)
    print(f"  💾 Saved model as: {model_filename}")

print("\n🎉 ΤΕΛΟΣ – Όλα τα μοντέλα εκπαιδεύτηκαν και σώθηκαν.")
