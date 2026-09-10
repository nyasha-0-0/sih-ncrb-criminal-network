import os
import numpy as np
import pandas as pd

def clean_and_export_dataset():
    # Load input dataset
    input_path = "synthetic_crime_15k.csv"
    if not os.path.exists(input_path):
        input_path = os.path.join("data", "synthetic_crime_15k.csv")
        
    print(f"[+] Loading dataset from: {os.path.abspath(input_path)}")
    df = pd.read_csv(input_path)
    print(f"[+] Raw Input Dataset Shape: {df.shape}")
    
    # 1. Pipeline Requirement 1: Drop Columns
    cols_to_drop = [
        'label_public_order_vs_law_and_order',
        'ipc_468_471',
        'cybercrime_indicator',
        'case_reference_id'
    ]
    
    df_pruned = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore').copy()
    print(f"[+] Dropped {len(cols_to_drop)} leakage & metadata columns: {cols_to_drop}")
    
    # 2. Pipeline Requirement 2: Separate Features and Targets
    target_cols = [
        'label_entity_role',
        'label_high_risk_cyber_syndicate',
        'label_bail_or_custody_relief',
        'label_conviction_outcome'
    ]
    
    targets_df = df_pruned[target_cols].copy()
    X_raw = df_pruned.drop(columns=target_cols).copy()
    
    # 3. Pipeline Requirement 3: Handle Remaining Categoricals (court_level, case_type)
    cat_cols = ['court_level', 'case_type']
    X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=False, dtype=int)
    
    # Combine cleaned features + targets for final dataset export
    cleaned_df = pd.concat([X_encoded, targets_df], axis=1)
    
    # 4. Pipeline Requirement 4: Validation & Correlation Checks (|r| <= 0.85)
    print("\n" + "="*65)
    print("      TARGET LEAKAGE CORRELATION AUDIT (|r| <= 0.85)")
    print("="*65)
    
    high_corr_violations = []
    for target in target_cols:
        for col in X_encoded.columns:
            # Skip non-numeric or boolean columns if any
            if np.issubdtype(X_encoded[col].dtype, np.number):
                valid_idx = X_encoded[col].notnull() & targets_df[target].notnull()
                if valid_idx.sum() > 0:
                    r = np.corrcoef(X_encoded[col][valid_idx], targets_df[target][valid_idx])[0, 1]
                    if abs(r) > 0.85:
                        high_corr_violations.append((col, target, r))
                        print(f"  [WARNING] High correlation: Feature '{col}' <-> Target '{target}' (r = {r:.4f})")
                        
    if len(high_corr_violations) == 0:
        print("[OK] AUDIT PASSED: Zero remaining features have |r| > 0.85 with any target variable!")
    else:
        print(f"[!] WARNING: Found {len(high_corr_violations)} features with |r| > 0.85")
        
    # Verify 0 leakage columns remain
    for col in cols_to_drop:
        assert col not in cleaned_df.columns, f"Assertion Error: Leakage column {col} still present!"
    print("[OK] ASSERTION PASSED: 0 leakage columns remain in the output dataframe.")
    
    # Export cleaned dataset
    output_filename = "synthetic_crime_15k_cleaned.csv"
    paths_to_export = [
        output_filename,
        os.path.join("data", output_filename)
    ]
    
    os.makedirs("data", exist_ok=True)
    for path in paths_to_export:
        cleaned_df.to_csv(path, index=False)
        print(f"[+] Successfully exported cleaned dataset to: {os.path.abspath(path)}")
        
    print("="*65)
    print(f"Final Cleaned Dataset Shape : {cleaned_df.shape}")
    print(f"Total Feature Count (X)     : {X_encoded.shape[1]}")
    print(f"Total Target Count (y)      : {targets_df.shape[1]}")
    print("="*65)
    
    return cleaned_df

if __name__ == "__main__":
    clean_and_export_dataset()
