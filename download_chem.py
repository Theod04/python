import pandas as pd
from chembl_webresource_client.new_client import new_client

TARGETS = {
    "AChE": "CHEMBL220",
    "BACE1": "CHEMBL4822",
    "ADAM10": "CHEMBL3650",
    "LRP1": "CHEMBL4296",
    "NMDA": "CHEMBL310",
    "SIRT3": "CHEMBL1741198",
    "MAO-A": "CHEMBL203",
    "MAO-B": "CHEMBL205",
    "MARK": "CHEMBL1913",
    "GLP-1": "CHEMBL256",
    "GSK-3B": "CHEMBL262",
    "ABCA1": "CHEMBL4302",
    "TREM2": "CHEMBL2134021",
    "CSF1R": "CHEMBL4799",
}

def download_target(name, chembl_id):
    if not chembl_id.startswith("CHEMBL"):
        print(f"[{name}] ➜ Παράλειψη (δεν έχεις βάλει σωστό CHEMBL ID)")
        return

    print(f"\n=== {name} ({chembl_id}) ===")
    activity = new_client.activity

    res = activity.filter(target_chembl_id=chembl_id).only(
        [
            "canonical_smiles",
            "standard_type",
            "standard_relation",
            "standard_value",
            "standard_units",
            "pchembl_value",
            "assay_type",
        ]
    )

    df = pd.DataFrame(res)

    if df.empty:
        print(f"[{name}] Δεν βρέθηκαν δεδομένα.")
        return

    if "canonical_smiles" not in df.columns:
        print(f"[{name}] Προσοχή: δεν υπάρχει στήλη 'canonical_smiles'. Σώζω ό,τι ήρθε.")
    else:
        df = df.dropna(subset=["canonical_smiles"])

    out_name = f"{name}.csv"
    df.to_csv(out_name, index=False)
    print(f"[{name}] Αποθηκεύτηκε: {out_name} (rows = {len(df)})")


if __name__ == "__main__":
    for protein, chembl_id in TARGETS.items():
        download_target(protein, chembl_id)

    print("\n✅ ΤΕΛΟΣ – Τα .csv είναι στον ίδιο φάκελο με το script.")
