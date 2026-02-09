import zipfile
import pickle
import os
import gc
import pandas as pd
import neurokit2 as nk
from tqdm import tqdm
from pathlib import Path

# --- Setup Paths ---
# This ensures we always look in the same folder as the script
BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / 'WESAD.zip'
OUTPUT_FILE = BASE_DIR / 'wesad_global_features.csv'
TEMP_DIR = BASE_DIR / 'temp_extraction'

# Create temp dir if missing
TEMP_DIR.mkdir(exist_ok=True)

subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
fs = 700
window_size = 60 * fs

def process_subject(subject_id, zip_ref):
    # Find the exact path inside the zip (handles case-sensitivity)
    all_files = zip_ref.namelist()
    # Find a file that ends with /S2.pkl or S2.pkl
    target_match = [f for f in all_files if f.endswith(f'{subject_id}.pkl')]
    
    if not target_match:
        print(f"Skipping {subject_id}: Not found in zip.")
        return
    
    internal_pkl_path = target_match[0]
    
    # Extract and get the ACTUAL path on your drive
    extracted_path = Path(zip_ref.extract(internal_pkl_path, path=TEMP_DIR))
    
    # Load Data
    with open(extracted_path, 'rb') as f:
        raw_data = pickle.load(f, encoding='latin1')
    
    labels = raw_data['label'].flatten()
    ecg = raw_data['signal']['chest']['ECG'].flatten()
    del raw_data
    gc.collect()

    # Process Windows (1=Baseline/0, 2=Stress/1)
    for label_val, mapped_label in [(1, 0), (2, 1)]:
        condition_ecg = ecg[labels == label_val]
        num_windows = len(condition_ecg) // window_size
        
        for i in range(num_windows):
            chunk = condition_ecg[i*window_size : (i+1)*window_size]
            try:
                signals, _ = nk.ecg_process(chunk, sampling_rate=fs)
                hrv = nk.hrv_time(signals, sampling_rate=fs)
                hrv['Subject_ID'] = subject_id
                hrv['Label'] = mapped_label
                
                # Append to CSV
                header = not OUTPUT_FILE.exists()
                hrv.to_csv(OUTPUT_FILE, mode='a', index=False, header=header)
            except:
                continue 

    # Safe Cleanup: Remove the file we just extracted
    if extracted_path.exists():
        os.remove(extracted_path)

# --- Execution ---
if not ZIP_PATH.exists():
    print(f"ERROR: Cannot find {ZIP_PATH}. Put the zip in {BASE_DIR}")
else:
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for s in tqdm(subjects, desc="Global Progress"):
            process_subject(s, z)
    print(f"\n✅ All done! Data saved to: {OUTPUT_FILE}")
