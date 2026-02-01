import pandas as pd

# Try to load the file and print just the column names
try:
    df = pd.read_csv('gen_ai.csv')
    print("--- COLUMNS FOUND IN GEN_AI.CSV ---")
    print(list(df.columns))
    print("-----------------------------------")
except Exception as e:
    print(f"Error reading file: {e}")