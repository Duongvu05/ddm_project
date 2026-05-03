import subprocess

print("Creating base EDA notebook...")
subprocess.run(["python", "notebooks/create_eda.py"], check=True)

print("Appending extra sections...")
subprocess.run(["python", "notebooks/create_eda_extra.py"], check=True)

print("Done.")