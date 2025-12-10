import requests
import pandas as pd

# Target: AChE = CHEMBL220
BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
TARGET = "CHEMBL220"

all_records = []
limit = 1000
offset = 0

while True:
    url = f"{BASE_URL}?target_chembl_id={TARGET}&limit={limit}&offset={offset}"
    print("Downloading:", url)

    response = requests.get(url)
    if response.status_code != 200:
        print("Error:", response.status_code)
        break

    data = response.json()

    # Αν δεν υπάρχουν άλλα αποτελέσματα → σταμάτα
    if "activities" not in data or len(data["activities"]) == 0:
        break

    all_records.extend(data["activities"])
    offset += limit

# Μετατροπή σε DataFrame
df = pd.json_normalize(all_records)

# Αποθήκευση CSV
df.to_csv("AChE_bioactivities.csv", index=False)
print("✔ CSV saved: AChE_bioactivities.csv")
print("✔ Rows:", len(df))
