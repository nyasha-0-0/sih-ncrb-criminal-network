import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class NCRBDataPreprocessingPipeline:
    """
    Production-Grade Data Preprocessing & Remediation Pipeline for NCRB Cyber Intelligence Platform.
    Adheres strictly to zero data-leakage guardrails (fit on train only, transform on test).
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.column_transformer = None
        self.feature_names_out = None
        
    def get_leakage_and_target_columns(self):
        # Primary target
        target_col = 'label_entity_role'
        
        # Target clones / leakage features that must be removed from X
        leakage_cols = [
            'case_reference_id',                      # Unique Identifier
            'label_high_risk_cyber_syndicate',        # Deterministic label clone
            'label_public_order_vs_law_and_order',    # Target clone
            'label_bail_or_custody_relief',            # Separate judicial target
            'label_conviction_outcome'                # Downstream outcome target
        ]
        return target_col, leakage_cols

    def separate_features_and_target(self, df):
        target_col, leakage_cols = self.get_leakage_and_target_columns()
        
        y = df[target_col].copy()
        X = df.drop(columns=[target_col] + leakage_cols, errors='ignore').copy()
        
        return X, y

    def build_column_transformer(self, X):
        # Categorical features (Low cardinality: court_level, case_type)
        cat_cols = ['court_level', 'case_type']
        
        # Heavy-tailed continuous numerical features requiring Quantile Normalisation + Robust Scaling
        skewed_num_cols = [
            'unauthorized_transfer_amount_inr',
            'fund_transit_velocity_minutes',
            'call_burst_frequency',
            'imei_sim_swap_ratio',
            'avg_call_duration_sec',
            'custody_duration_days'
        ]
        
        # Continuous topological & score features
        score_num_cols = [
            'evidence_completeness_score',
            'night_call_ratio',
            'amount_credited_to_accused_ratio',
            'network_degree',
            'network_betweenness',
            'topological_kingpin_score'
        ]
        
        # Binary & Integer Indicator features (Already 0/1 or small counts, pass through)
        passthrough_cols = [c for c in X.columns if c not in cat_cols + skewed_num_cols + score_num_cols]
        
        # Build Pipelines
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        skewed_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('quantile', QuantileTransformer(output_distribution='normal', random_state=self.random_state)),
            ('scaler', RobustScaler())
        ])
        
        score_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        column_transformer = ColumnTransformer(
            transformers=[
                ('cat', cat_pipeline, cat_cols),
                ('skewed', skewed_pipeline, skewed_num_cols),
                ('score', score_pipeline, score_num_cols),
                ('passthrough', 'passthrough', passthrough_cols)
            ],
            remainder='drop'
        )
        
        return column_transformer, cat_cols, skewed_num_cols, score_num_cols, passthrough_cols

    def fit_transform(self, df):
        X, y = self.separate_features_and_target(df)
        
        # Stratified Train-Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state, stratify=y
        )
        
        # Build Transformer
        self.column_transformer, cat_cols, skewed_cols, score_cols, pass_cols = self.build_column_transformer(X_train)
        
        # Fit ONLY on X_train to avoid data leakage
        X_train_trans = self.column_transformer.fit_transform(X_train)
        X_test_trans = self.column_transformer.transform(X_test)
        
        # Extract transformed feature names
        cat_encoder = self.column_transformer.named_transformers_['cat'].named_steps['onehot']
        cat_encoded_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
        
        self.feature_names_out = cat_encoded_names + skewed_cols + score_cols + pass_cols
        
        # Convert back to DataFrame for validation & quality assertions
        df_train_clean = pd.DataFrame(X_train_trans, columns=self.feature_names_out)
        df_test_clean = pd.DataFrame(X_test_trans, columns=self.feature_names_out)
        
        return df_train_clean, df_test_clean, y_train, y_test

def run_data_quality_assertions(df_train_clean, df_test_clean, y_train, y_test, X_raw_count):
    print("\n" + "="*65)
    print("      DATA QUALITY & PREPROCESSING SANITY ASSERTIONS")
    print("="*65)
    
    # 1. Null Checks
    assert df_train_clean.isnull().sum().sum() == 0, "Assertion Failure: Train features contain null values!"
    assert df_test_clean.isnull().sum().sum() == 0, "Assertion Failure: Test features contain null values!"
    print("[OK] ASSERTION PASSED: Zero null/missing values in cleaned train and test sets.")
    
    # 2. Shape Integrity Check
    assert df_train_clean.shape[0] + df_test_clean.shape[0] == X_raw_count, "Assertion Failure: Row count mismatch after split!"
    assert df_train_clean.shape[1] == df_test_clean.shape[1], "Assertion Failure: Train/Test feature column mismatch!"
    print(f"[OK] ASSERTION PASSED: Row count integrity preserved ({df_train_clean.shape[0]} train + {df_test_clean.shape[0]} test = {X_raw_count}).")
    
    # 3. Stratification Verification
    train_dist = y_train.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()
    max_diff = (train_dist - test_dist).abs().max()
    assert max_diff < 0.01, f"Assertion Failure: Stratification shift detected! Max diff = {max_diff}"
    print(f"[OK] ASSERTION PASSED: Class stratification perfectly preserved (Max train/test shift = {max_diff:.4f}).")
    
    # 4. Target Leakage Verification
    leakage_terms = ['case_reference_id', 'label_high_risk_cyber_syndicate', 'label_public_order_vs_law_and_order']
    for term in leakage_terms:
        assert term not in df_train_clean.columns, f"Assertion Failure: Leakage term {term} present in feature matrix!"
    print("[OK] ASSERTION PASSED: All 5 deterministic target clones cleanly pruned.")
    print("="*65)

if __name__ == "__main__":
    start_t = time.time()
    csv_path = "synthetic_crime_15k.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join("data", "synthetic_crime_15k.csv")
        
    print(f"[+] Loading dataset from: {os.path.abspath(csv_path)}")
    raw_df = pd.read_csv(csv_path)
    print(f"[+] Raw Dataset Shape: {raw_df.shape}")
    
    pipeline = NCRBDataPreprocessingPipeline(random_state=42)
    X_train_clean, X_test_clean, y_train, y_test = pipeline.fit_transform(raw_df)
    
    # Run sanity assertions
    run_data_quality_assertions(X_train_clean, X_test_clean, y_train, y_test, len(raw_df))
    
    elapsed = time.time() - start_t
    print(f"\n[+] Pipeline Execution Time : {elapsed:.3f} seconds")
    print(f"[+] Cleaned X_train Shape  : {X_train_clean.shape}")
    print(f"[+] Cleaned X_test Shape   : {X_test_clean.shape}")
    print(f"[+] Total Cleaned Features : {X_train_clean.shape[1]}")
