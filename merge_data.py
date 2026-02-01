import pandas as pd
import os

print("--- STARTING FINAL ROBUST MERGE ---")

# 1. Initialize a master list to hold all data
all_dfs = []

# ==========================================
# PART 1: LOAD NEW DATA (2025/2026 Context)
# ==========================================

# --- 1.1 Custom Data (Your manual fixes) ---
try:
    print("\n1. Loading custom_2025.csv...")
    if os.path.exists('custom_2025.csv'):
        df_custom = pd.read_csv('custom_2025.csv')
        # Boost this data so the model prioritizes it
        df_custom = pd.concat([df_custom] * 100, ignore_index=True)
        
        if 'label' not in df_custom.columns:
            df_custom['label'] = 0 # Default if missing
            
        df_custom = df_custom[['text', 'label']]
        all_dfs.append(df_custom)
        print(f"✅ Loaded {len(df_custom)} rows from Custom Data")
    else:
        print("⚠️ custom_2025.csv not found. Skipping.")
except Exception as e:
    print(f"❌ Error loading custom data: {e}")

# --- 1.2 India News Headlines (Real News) ---
try:
    print("\n2. Loading india-news-headlines.csv...")
    if os.path.exists('india-news-headlines.csv'):
        # This file is usually massive (3M rows). Loading just the newest 50,000 to save RAM.
        # If you have 32GB+ RAM, remove 'nrows=50000'
        df_real = pd.read_csv('india-news-headlines.csv') 
        
        # This specific dataset usually has a column 'headline_text'
        if 'headline_text' in df_real.columns:
            df_real.rename(columns={'headline_text': 'text'}, inplace=True)
        elif 'content' in df_real.columns:
            df_real.rename(columns={'content': 'text'}, inplace=True)
            
        # Filter: Only keep rows that have text
        df_real = df_real.dropna(subset=['text'])
        
        # Set Label to 1 (Real)
        df_real['label'] = 1 
        
        # Keep only necessary columns
        df_real = df_real[['text', 'label']]
        
        # OPTIONAL: If the file is 3 Million rows, take only the last 50,000 (Newest)
        # df_real = df_real.tail(50000) 
        
        all_dfs.append(df_real)
        print(f"✅ Loaded {len(df_real)} rows from India Headlines")
    else:
        print("⚠️ india-news-headlines.csv not found. Skipping.")
except Exception as e:
    print(f"❌ Error loading India News: {e}")

# --- 1.3 Gen AI Fake News ---
try:
    print("\n3. Loading gen_ai.csv...")
    if os.path.exists('gen_ai.csv'):
        df_gen = pd.read_csv('gen_ai.csv')
        
        if 'misinformation' in df_gen.columns:
            df_gen.rename(columns={'misinformation': 'text'}, inplace=True)
        
        df_gen['label'] = 0 # Force Fake
        df_gen = df_gen[['text', 'label']]
        all_dfs.append(df_gen)
        print(f"✅ Loaded {len(df_gen)} rows from Gen AI Data")
    else:
        print("⚠️ gen_ai.csv not found. Skipping.")
except Exception as e:
    print(f"❌ Error loading Gen AI data: {e}")

# ==========================================
# PART 2: LOAD OLD DATA (Kaggle Dataset)
# ==========================================

# --- 2.1 True.csv ---
try:
    print("\n4. Loading True.csv...")
    if os.path.exists('True.csv'):
        df_true = pd.read_csv('True.csv')
        df_true['label'] = 1
        if 'title' in df_true.columns:
            df_true['text'] = df_true['title'] + " " + df_true['text']
        
        all_dfs.append(df_true[['text', 'label']])
        print(f"✅ Loaded {len(df_true)} rows from True.csv")
    else:
        print("⚠️ True.csv not found. Skipping.")
except Exception as e:
    print(f"❌ Error loading True.csv: {e}")

# --- 2.2 Fake.csv ---
try:
    print("\n5. Loading Fake.csv...")
    if os.path.exists('Fake.csv'):
        df_fake_old = pd.read_csv('Fake.csv')
        df_fake_old['label'] = 0
        if 'title' in df_fake_old.columns:
            df_fake_old['text'] = df_fake_old['title'] + " " + df_fake_old['text']
        
        all_dfs.append(df_fake_old[['text', 'label']])
        print(f"✅ Loaded {len(df_fake_old)} rows from Fake.csv")
    else:
        print("⚠️ Fake.csv not found. Skipping.")
except Exception as e:
    print(f"❌ Error loading Fake.csv: {e}")

# --- 2.3 Bharat.csv ---
try:
    print("\n6. Loading bharat.csv...")
    if os.path.exists('bharat.csv'):
        df_bharat = pd.read_csv('bharat.csv')
        df_bharat.columns = df_bharat.columns.str.strip().str.lower()
        
        # Rename label column
        if 'fake' in df_bharat.columns:
            df_bharat.rename(columns={'fake': 'label'}, inplace=True)
        elif 'class' in df_bharat.columns:
            df_bharat.rename(columns={'class': 'label'}, inplace=True)
            
        # Map labels
        if 'label' in df_bharat.columns:
            mapping = {'TRUE': 1, 'True': 1, 1: 1, 'FAKE': 0, 'Fake': 0, 0: 0}
            df_bharat['label'] = df_bharat['label'].map(mapping)
            df_bharat.dropna(subset=['label'], inplace=True) # Drop if mapping failed
            all_dfs.append(df_bharat[['text', 'label']])
            print(f"✅ Loaded {len(df_bharat)} rows from bharat.csv")
    else:
        print("⚠️ bharat.csv not found. Skipping.")
except Exception as e:
    print(f"❌ Error loading bharat.csv: {e}")

# ==========================================
# PART 3: FINAL MERGE
# ==========================================

print("\n--- MERGING DATA ---")
if all_dfs:
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Clean
    master_df.dropna(subset=['text', 'label'], inplace=True)
    master_df['label'] = master_df['label'].astype(int)
    
    # Save
    master_df.to_csv('final_master_dataset.csv', index=False)
    print(f"🎉 DONE! Created 'final_master_dataset.csv' with {len(master_df)} articles.")
    print("👉 Now run 'python train_model.py'")
else:
    print("❌ NO DATA LOADED. Please check that your CSV files are in the folder.")