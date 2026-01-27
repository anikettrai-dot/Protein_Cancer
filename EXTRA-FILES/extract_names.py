import pandas as pd

# Load the OncoKB file
try:
    # Try loading the file
    df = pd.read_csv("cancerGeneList.tsv", sep="\t")
    
    # The gene names are usually in the 'Hugo Symbol' column
    if 'Hugo Symbol' in df.columns:
        # Save just the names to a text file
        df['Hugo Symbol'].to_csv("genes_to_map.txt", index=False, header=False)
        print("SUCCESS! Created genes_to_map.txt")
        print(f"Found {len(df)} cancer genes.")
    else:
        print("ERROR: Could not find 'Hugo Symbol' column.")
        print("Columns found:", df.columns)

except Exception as e:
    print(f"ERROR reading file: {e}")