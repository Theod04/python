import os

for f in os.listdir('.'):
    if f.endswith('.joblib'):
        print(f, os.path.getsize(f))
