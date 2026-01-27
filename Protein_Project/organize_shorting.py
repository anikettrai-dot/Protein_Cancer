import os
import shutil
import pandas as pd
from tqdm import tqdm

# --- CONFIG ---
# 1. Where your images are NOW (mixed together)
IMAGE_DIR = r"D:\Protein_Cancer\Protein_Project\dataset_images"

# 2. Your Mapping File (Tells us which ID is Cancer)
MAP_FILE = r"D:\Protein_Cancer\Protein_Project\final_mapping.csv"
# --------------

def organize_data():
    print(f"--- 📂 Organizing Dataset in {IMAGE_DIR} ---")
    
    # Check paths
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Error: Image folder not found!")
        return
    if not os.path.exists(MAP_FILE):
        print(f"❌ Error: Mapping file not found!")
        return

    # Create Subfolders
    cancer_path = os.path.join(IMAGE_DIR, "Cancer")
    normal_path = os.path.join(IMAGE_DIR, "Non_Cancer")
    
    os.makedirs(cancer_path, exist_ok=True)
    os.makedirs(normal_path, exist_ok=True)

    # Read Cancer IDs
    try:
        df = pd.read_csv(MAP_FILE)
        # Get IDs from 2nd column ('ID'), strip whitespace
        cancer_ids = set(df.iloc[:, 1].astype(str).str.strip())
        print(f"📋 Loaded {len(cancer_ids)} Cancer IDs.")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Move Files
    files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]
    print(f"🔄 Sorting {len(files)} images...")
    
    c_count = 0
    n_count = 0
    
    for f in tqdm(files):
        src = os.path.join(IMAGE_DIR, f)
        
        # Check if filename matches a cancer ID
        # e.g., "AF-P04637-F1.png" contains "P04637"
        is_cancer = False
        for cid in cancer_ids:
            if cid in f:
                is_cancer = True
                break
        
        if is_cancer:
            dst = os.path.join(cancer_path, f)
            shutil.move(src, dst)
            c_count += 1
        else:
            dst = os.path.join(normal_path, f)
            shutil.move(src, dst)
            n_count += 1
            
    print("-" * 30)
    print("✅ SORTING COMPLETE!")
    print(f"🦠 Cancer:     {c_count}")
    print(f"🟢 Non-Cancer: {n_count}")

if __name__ == "__main__":
    organize_data()