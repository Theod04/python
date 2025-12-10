import requests

targets = {
    "AChE": "CHEMBL220",
    "BACE1": "CHEMBL4822",
    "MAO_A": "CHEMBL203",
    "MAO_B": "CHEMBL205",
    "GSK3B": "CHEMBL262",
    "ADAM10": "CHEMBL3650",
    "MARK2": "CHEMBL1913",
    "GLP1R": "CHEMBL256",
    "SIRT3": "CHEMBL1741198",
    "ABCA1": "CHEMBL4302",
    "CSF1R": "CHEMBL4799",
    "NMDA": "CHEMBL310",
}

for name, chembl_id in targets.items():
    url = f"https://www.ebi.ac.uk/chembl/api/data/activity.csv?target_chembl_id={chembl_id}"
    print(f"Downloading {name}...")
    r = requests.get(url)
    with open(f"{name}.csv", "wb") as f:
        f.write(r.content)

print("Done!")

