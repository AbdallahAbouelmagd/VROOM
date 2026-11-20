import pandas as pd
import glob
import os

files = glob.glob(os.path.join(os.path.dirname(__file__), "*.csv"))

training = 0
testing = 0

for f in files:
    df = pd.read_csv(f)
    training += (df["data_type"] == "training").sum()
    testing += (df["data_type"] == "testing").sum()

print("training:", training)
print("testing:", testing)
print("total:", training + testing)