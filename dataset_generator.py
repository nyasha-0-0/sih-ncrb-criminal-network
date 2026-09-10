import os
import time
import numpy as np
import pandas as pd

def generate_synthetic_dataset(N=12500, seed=42):
    start_time = time.time()
    np.random.seed(seed)

    # FIX 1: Balanced Archetype Proportions to avoid Anomaly Inversion Trap
    # 0: Street Snatching (~20%)
    # 1: ATM Distraction (~15%)
    # 2: Fake Call Center / Phishing (~20%)
    # 3: Telecom / Parallel VoIP Exchange (~15%)
    # 4: Core Banking Hacking & Mule (~15%)
    # 5: Civilian / Legitimate / Dismissed (~15% - increased from 6% to fix anomaly inversion)
    
    probs = [0.20, 0.15, 0.20, 0.15, 0.15, 0.15]
    archetype = np.random.choice(6, size=N, p=probs)
    
    # Case reference IDs
    case_ref_id = np.array([f"NCRB-SYNTH-{100001 + i}" for i in range(N)])
    
    # Mask definitions
    mask0 = (archetype == 0)
    mask1 = (archetype == 1)
    mask2 = (archetype == 2)
    mask3 = (archetype == 3)
    mask4 = (archetype == 4)
    mask5 = (archetype == 5)

    # Categorical fields
    court_levels = np.empty(N, dtype=object)
    case_types = np.empty(N, dtype=object)
    
    court_choices = ["Trial Court", "High Court", "Supreme Court"]
    case_type_choices = ["Writ Petition (Detention)", "Sec 482 CrPC (Quashing)", "Sec 439 CrPC (Regular Bail)", "Trial Judgment"]
    
    court_levels[mask0] = np.random.choice(court_choices, size=mask0.sum(), p=[0.80, 0.20, 0.0])
    case_types[mask0] = np.random.choice(case_type_choices, size=mask0.sum(), p=[0.25, 0.25, 0.35, 0.15])

    court_levels[mask1] = np.random.choice(court_choices, size=mask1.sum(), p=[0.60, 0.38, 0.02])
    case_types[mask1] = np.random.choice(case_type_choices, size=mask1.sum(), p=[0.40, 0.20, 0.30, 0.10])

    court_levels[mask2] = np.random.choice(court_choices, size=mask2.sum(), p=[0.50, 0.45, 0.05])
    case_types[mask2] = np.random.choice(case_type_choices, size=mask2.sum(), p=[0.10, 0.20, 0.60, 0.10])

    court_levels[mask3] = np.random.choice(court_choices, size=mask3.sum(), p=[0.10, 0.70, 0.20])
    case_types[mask3] = np.random.choice(case_type_choices, size=mask3.sum(), p=[0.10, 0.30, 0.55, 0.05])

    court_levels[mask4] = np.random.choice(court_choices, size=mask4.sum(), p=[0.40, 0.50, 0.10])
    case_types[mask4] = np.random.choice(case_type_choices, size=mask4.sum(), p=[0.05, 0.25, 0.60, 0.10])

    court_levels[mask5] = np.random.choice(court_choices, size=mask5.sum(), p=[0.80, 0.20, 0.0])
    case_types[mask5] = np.random.choice(case_type_choices, size=mask5.sum(), p=[0.10, 0.60, 0.20, 0.10])

    # Alleged cases & accused
    alleged_cases_pool = np.array([1, 2, 4, 7, 15, 21, 27, 45])
    number_of_cases_alleged = np.random.choice(alleged_cases_pool, size=N, p=[0.50, 0.20, 0.12, 0.08, 0.04, 0.03, 0.02, 0.01])
    number_of_cases_alleged[mask4] = np.random.choice(alleged_cases_pool, size=mask4.sum(), p=[0.10, 0.25, 0.25, 0.20, 0.10, 0.05, 0.03, 0.02])

    number_of_accused = np.random.randint(1, 6, size=N)
    number_of_accused[mask2 | mask3 | mask4] = np.random.randint(3, 26, size=(mask2 | mask3 | mask4).sum())

    victim_count = np.random.randint(1, 3, size=N)
    victim_count[mask2 | mask4] = np.random.randint(5, 101, size=(mask2 | mask4).sum())
    victim_count[mask3] = np.random.randint(10, 80, size=mask3.sum())

    # Indicators
    cybercrime_indicator = np.where(mask2 | mask3 | mask4, 1, np.where(mask5, np.random.binomial(1, 0.10, size=N), 0))
    financial_fraud_indicator = np.where(mask1 | mask2 | mask3 | mask4, 1, np.where(mask5, np.random.binomial(1, 0.05, size=N), 0))
    property_offence_indicator = np.where(mask0 | mask1, 1, 0)
    organized_operation_indicator = np.where(mask2 | mask3 | mask4, 1, np.where(mask0, np.random.binomial(1, 0.25, size=N), np.where(mask1, np.random.binomial(1, 0.60, size=N), 0)))
    repeat_offender_indicator = np.where(mask5, 0, np.random.binomial(1, 0.35, size=N))
    social_engineering_indicator = np.where(mask1 | mask2, 1, 0)

    cross_state_indicator = np.where(mask2 | mask3 | mask4, np.random.binomial(1, 0.88, size=N), np.where(mask5, np.random.binomial(1, 0.05, size=N), np.random.binomial(1, 0.08, size=N)))
    cross_border_indicator = np.where(mask3, np.random.binomial(1, 0.25, size=N), np.where(mask2, np.random.binomial(1, 0.03, size=N), np.where(mask4, np.random.binomial(1, 0.05, size=N), 0)))

    # Penal Sections
    ipc_379 = np.where(mask0 | mask1, 1, 0)
    ipc_420 = np.where(mask1 | mask2 | mask3 | mask4, 1, np.where(mask5, 0, np.random.binomial(1, 0.1, size=N)))
    ipc_468_471 = np.where(mask2 | mask3 | mask4, 1, 0)
    ipc_120b_or_34 = np.where(mask2 | mask3 | mask4, 1, np.where(mask0, np.random.binomial(1, 0.4, size=N), np.where(mask1, np.random.binomial(1, 0.7, size=N), 0)))

    # FIX 3: Continuous Gaussian Jitter to eliminate hard synthetic cutoffs
    evidence_completeness_score = np.random.uniform(0.4, 0.95, size=N) + np.random.normal(0, 0.03, size=N)
    evidence_completeness_score[mask5] = np.random.uniform(0.1, 0.4, size=mask5.sum()) + np.random.normal(0, 0.02, size=mask5.sum())
    evidence_completeness_score = np.clip(evidence_completeness_score, 0.05, 0.99)

    charge_sheet_filed = np.random.binomial(1, 0.85, size=N)
    charge_sheet_filed[mask3] = 1
    charge_sheet_filed[mask5] = np.random.binomial(1, 0.30, size=mask5.sum())

    # FIX 3: Add continuous jitter to custody_duration_days
    base_custody = np.random.randint(15, 180, size=N).astype(float)
    base_custody[mask3] = np.random.randint(120, 300, size=mask3.sum()).astype(float)
    base_custody[mask5] = np.random.randint(1, 45, size=mask5.sum()).astype(float)
    custody_duration_days = np.clip(base_custody + np.random.normal(0, 4.0, size=N), 1.0, 365.0).astype(int)

    statutory_cert_present = np.where(mask2 | mask3 | mask4, np.random.binomial(1, 0.80, size=N), np.random.binomial(1, 0.35, size=N))
    chain_of_custody_complete = np.where(statutory_cert_present == 1, np.random.binomial(1, 0.85, size=N), np.random.binomial(1, 0.20, size=N))

    # Telecom Cluster
    fake_call_center_indicator = np.where(mask2, 1, 0)
    parallel_telephone_exchange_indicator = np.where(mask3, 1, 0)
    bulk_corporate_sim_indicator = np.where(mask2, np.random.binomial(1, 0.75, size=N), np.where(mask3, 1, 0))
    forged_company_indicator = np.where(mask2, np.random.binomial(1, 0.65, size=N), np.where(mask3, 1, 0))
    voip_calling_indicator = np.where(mask3, 1, np.where(mask2, np.random.binomial(1, 0.70, size=N), 0))
    government_telecom_revenue_loss = np.where(mask3, 1, 0)
    vishing_indicator = np.where(mask2, 1, 0)

    # FIX 3: Add smooth Gaussian noise to night_call_ratio
    night_call_ratio = np.random.normal(0.08, 0.04, size=N)
    night_call_ratio[mask2 | mask3] = np.random.normal(0.68, 0.12, size=(mask2 | mask3).sum())
    night_call_ratio[mask5] = np.random.normal(0.05, 0.03, size=mask5.sum())
    night_call_ratio = np.clip(night_call_ratio + np.random.normal(0, 0.02, size=N), 0.0, 1.0)

    call_burst_frequency = np.random.randint(1, 10, size=N)
    call_burst_frequency[mask2] = np.random.randint(40, 151, size=mask2.sum())
    call_burst_frequency[mask3] = np.random.randint(100, 501, size=mask3.sum())

    imei_sim_swap_ratio = np.random.uniform(1.0, 1.2, size=N)
    imei_sim_swap_ratio[mask2] = np.random.uniform(3.0, 8.0, size=mask2.sum())
    imei_sim_swap_ratio[mask3] = np.random.uniform(3.5, 14.0, size=mask3.sum())
    imei_sim_swap_ratio = np.clip(imei_sim_swap_ratio + np.random.normal(0, 0.1, size=N), 1.0, 20.0)

    avg_call_duration_sec = np.random.uniform(30.0, 180.0, size=N)
    avg_call_duration_sec[mask2] = np.random.uniform(45.0, 300.0, size=mask2.sum())
    avg_call_duration_sec[mask3] = np.random.uniform(15.0, 90.0, size=mask3.sum())

    multiple_calling_numbers_indicator = np.where(mask2 | mask3, 1, 0)
    it_act_66d = np.where(mask2 | mask3, 1, np.where(mask1, np.random.binomial(1, 0.3, size=N), 0))
    telegraph_act_sections = np.where(mask3, 1, 0)

    # Core Banking Cluster
    internet_banking_software_targeted = np.where(mask4, 1, 0)
    banking_password_compromised = np.where(mask4, 1, 0)

    # FIX 2: Replace -1.0 placeholder with np.nan and add explicit boolean flag
    is_fund_transit_velocity_applicable = np.where(mask2 | mask4, 1, 0)
    fund_transit_velocity_minutes = np.full(N, np.nan, dtype=np.float64)
    fund_transit_velocity_minutes[mask2 | mask4] = np.random.exponential(scale=10.0, size=(mask2 | mask4).sum()) + 1.0

    unauthorized_transfer_amount_inr = np.zeros(N, dtype=np.float64)
    unauthorized_transfer_amount_inr[mask0] = np.random.uniform(3000.0, 45000.0, size=mask0.sum())
    unauthorized_transfer_amount_inr[mask1] = np.random.uniform(15000.0, 150000.0, size=mask1.sum())
    unauthorized_transfer_amount_inr[mask2] = np.random.uniform(250000.0, 4500000.0, size=mask2.sum())
    unauthorized_transfer_amount_inr[mask3] = np.random.uniform(50000.0, 2500000.0, size=mask3.sum())
    unauthorized_transfer_amount_inr[mask4] = np.random.uniform(500000.0, 15000000.0, size=mask4.sum())

    amount_credited_to_accused_ratio = np.random.uniform(0.0, 0.2, size=N)
    amount_credited_to_accused_ratio[mask4] = np.random.uniform(0.15, 1.0, size=mask4.sum())

    account_frozen_indicator = np.where(mask4, 1, np.where(mask2, np.random.binomial(1, 0.6, size=N), 0))
    suspicious_credit_not_reported = np.where(mask4, 1, 0)
    bank_correction_request_absent = np.where(mask4, 1, 0)

    fan_in_count = np.zeros(N, dtype=int)
    fan_in_count[mask4] = np.random.randint(3, 26, size=mask4.sum())

    fraudulent_website_count = np.zeros(N, dtype=int)
    fraudulent_website_count[mask2] = np.random.randint(1, 6, size=mask2.sum())

    payment_gateway_count = np.zeros(N, dtype=int)
    payment_gateway_count[mask2] = np.random.randint(1, 5, size=mask2.sum())

    linked_beneficiary_account_count = np.zeros(N, dtype=int)
    linked_beneficiary_account_count[mask2] = np.random.randint(2, 11, size=mask2.sum())
    linked_beneficiary_account_count[mask4] = np.random.randint(3, 16, size=mask4.sum())

    card_credential_harvesting_indicator = np.where(mask2, np.random.binomial(1, 0.80, size=N), 0)
    it_act_66 = np.where(mask4, 1, 0)
    it_act_66c = np.where(mask2 | mask4, 1, 0)

    # Physical Theft Cluster
    vehicle_assisted_theft = np.where(mask0, np.random.binomial(1, 0.85, size=N), 0)
    helmet_concealment_indicator = np.where(mask0, np.random.binomial(1, 0.70, size=N), 0)
    victim_walking_alone_indicator = np.where(mask0, np.random.binomial(1, 0.80, size=N), 0)
    victim_approached_from_behind = np.where(mask0, np.random.binomial(1, 0.85, size=N), 0)
    physical_force_used = np.where(mask0, np.random.binomial(1, 0.60, size=N), 0)

    stolen_physical_asset_count = np.zeros(N, dtype=int)
    stolen_physical_asset_count[mask0] = np.random.poisson(lam=1.5, size=mask0.sum()) + 1
    stolen_asset_recovered = np.where(mask0, np.random.binomial(1, 0.40, size=N), 0)

    victim_identification_success = np.zeros(N, dtype=int)
    victim_identification_success[mask0] = np.where(helmet_concealment_indicator[mask0] == 1, 0, np.random.binomial(1, 0.65, size=mask0.sum()))
    victim_identification_success[mask1 | mask2 | mask5] = np.random.binomial(1, 0.80, size=(mask1 | mask2 | mask5).sum())

    test_identification_parade_validity = np.where(victim_identification_success == 1, 1, 0)
    reasonable_doubt_identity_gap = np.where(victim_identification_success == 0, 1, 0)

    ipc_356 = np.where(mask0, np.random.binomial(1, 0.70, size=N), 0)
    ipc_392 = np.where(mask0, np.random.binomial(1, 0.40, size=N), 0)
    ipc_411 = np.where(mask0, np.random.binomial(1, 0.50, size=N), np.where(mask1, np.random.binomial(1, 0.60, size=N), 0))
    ipc_448 = np.where(mask0, np.random.binomial(1, 0.15, size=N), 0)
    ipc_380 = np.where(mask0, np.random.binomial(1, 0.20, size=N), 0)

    # Ground-Truth Roles
    label_entity_role = np.zeros(N, dtype=int)
    label_entity_role[mask0] = np.random.choice([2, 0], size=mask0.sum(), p=[0.85, 0.15])
    label_entity_role[mask1] = np.random.choice([2, 5, 4], size=mask1.sum(), p=[0.50, 0.30, 0.20])
    label_entity_role[mask2] = np.random.choice([5, 1, 3, 4], size=mask2.sum(), p=[0.40, 0.10, 0.20, 0.30])
    label_entity_role[mask3] = np.random.choice([3, 1, 4], size=mask3.sum(), p=[0.50, 0.25, 0.25])
    label_entity_role[mask4] = np.random.choice([4, 1, 3], size=mask4.sum(), p=[0.55, 0.20, 0.25])
    label_entity_role[mask5] = 0

    # FIX 4: Graph Centrality, Degree, and Topological Kingpin Score
    # Mastermind (1) -> Low Degree (1 to 3 connections), High Betweenness (0.42 to 0.95)
    # Mule (4) -> High Degree (12 to 45 connections), Moderate Betweenness (0.08 to 0.32)
    # Tech Operator (3) / Caller (5) -> Moderate Degree (4 to 12), Low Betweenness (0.05 to 0.25)
    # Civilian (0) -> Minimal Degree (1 to 2), Zero Betweenness (0.01 to 0.05)
    
    network_degree = np.random.randint(1, 5, size=N)
    network_betweenness = np.random.uniform(0.01, 0.10, size=N)

    is_kingpin_mask = (label_entity_role == 1)
    is_mule_mask = (label_entity_role == 4)
    is_operator_mask = (label_entity_role == 3) | (label_entity_role == 5)

    network_degree[is_kingpin_mask] = np.random.randint(1, 4, size=is_kingpin_mask.sum())
    network_betweenness[is_kingpin_mask] = np.random.uniform(0.42, 0.95, size=is_kingpin_mask.sum())

    network_degree[is_mule_mask] = np.random.randint(12, 46, size=is_mule_mask.sum())
    network_betweenness[is_mule_mask] = np.random.uniform(0.08, 0.32, size=is_mule_mask.sum())

    network_degree[is_operator_mask] = np.random.randint(4, 12, size=is_operator_mask.sum())
    network_betweenness[is_operator_mask] = np.random.uniform(0.05, 0.25, size=is_operator_mask.sum())

    # FIX 4 Formula: KingpinScore = Betweenness / log(1 + Degree)
    topological_kingpin_score = network_betweenness / np.log(1.0 + network_degree)

    label_high_risk_cyber_syndicate = np.where(mask2 | mask3 | mask4, 1, 0)
    label_public_order_vs_law_and_order = np.where(mask2 | mask3 | mask4, 1, 0)

    # Smooth Probabilistic Judgments with Noise (Fix 3)
    naushad_prob = 1.0 / (1.0 + np.exp(-(custody_duration_days - 145.0) / 12.0))
    label_bail_or_custody_relief = np.zeros(N, dtype=int)
    label_bail_or_custody_relief[mask5] = 1
    label_bail_or_custody_relief[mask0] = np.where(reasonable_doubt_identity_gap[mask0] == 1, 1, np.random.binomial(1, 0.4, size=mask0.sum()))
    label_bail_or_custody_relief[mask1] = 1

    mask3_naushad = mask3 & (charge_sheet_filed == 1)
    label_bail_or_custody_relief[mask3_naushad] = np.random.binomial(1, np.clip(naushad_prob[mask3_naushad], 0.1, 0.95))
    label_bail_or_custody_relief[mask3 & ~mask3_naushad] = np.random.binomial(1, 0.3, size=(mask3 & ~mask3_naushad).sum())

    walia_mask = mask4 & (account_frozen_indicator == 1) & (suspicious_credit_not_reported == 1) & (number_of_cases_alleged > 1)
    label_bail_or_custody_relief[walia_mask] = np.random.binomial(1, 0.08, size=walia_mask.sum())
    label_bail_or_custody_relief[mask4 & ~walia_mask] = np.random.binomial(1, 0.35, size=(mask4 & ~walia_mask).sum())
    label_bail_or_custody_relief[mask2] = np.random.binomial(1, 0.30, size=mask2.sum())

    label_conviction_outcome = np.zeros(N, dtype=int)
    label_conviction_outcome[mask0] = np.where(reasonable_doubt_identity_gap[mask0] == 1, 0, np.where(evidence_completeness_score[mask0] > 0.65, 1, 0))
    label_conviction_outcome[mask1] = np.where(evidence_completeness_score[mask1] > 0.70, 1, 0)
    label_conviction_outcome[mask2] = np.where((evidence_completeness_score[mask2] > 0.60) & (statutory_cert_present[mask2] == 1), 1, 0)
    label_conviction_outcome[mask3] = np.where((evidence_completeness_score[mask3] > 0.65) & (statutory_cert_present[mask3] == 1), 1, 0)
    label_conviction_outcome[mask4] = np.where((evidence_completeness_score[mask4] > 0.60) & (account_frozen_indicator[mask4] == 1), 1, 0)
    label_conviction_outcome[mask5] = 0

    df = pd.DataFrame({
        'archetype_id': archetype,
        'case_reference_id': case_ref_id,
        'court_level': court_levels,
        'case_type': case_types,
        'number_of_cases_alleged': number_of_cases_alleged,
        'number_of_accused': number_of_accused,
        'victim_count': victim_count,
        'cross_state_indicator': cross_state_indicator,
        'cross_border_indicator': cross_border_indicator,
        'cybercrime_indicator': cybercrime_indicator,
        'financial_fraud_indicator': financial_fraud_indicator,
        'property_offence_indicator': property_offence_indicator,
        'organized_operation_indicator': organized_operation_indicator,
        'repeat_offender_indicator': repeat_offender_indicator,
        'social_engineering_indicator': social_engineering_indicator,
        'ipc_379': ipc_379,
        'ipc_420': ipc_420,
        'ipc_468_471': ipc_468_471,
        'ipc_120b_or_34': ipc_120b_or_34,
        'evidence_completeness_score': np.round(evidence_completeness_score, 4),
        'charge_sheet_filed': charge_sheet_filed,
        'custody_duration_days': custody_duration_days,
        'statutory_cert_present': statutory_cert_present,
        'chain_of_custody_complete': chain_of_custody_complete,
        
        'fake_call_center_indicator': fake_call_center_indicator,
        'parallel_telephone_exchange_indicator': parallel_telephone_exchange_indicator,
        'bulk_corporate_sim_indicator': bulk_corporate_sim_indicator,
        'forged_company_indicator': forged_company_indicator,
        'voip_calling_indicator': voip_calling_indicator,
        'government_telecom_revenue_loss': government_telecom_revenue_loss,
        'vishing_indicator': vishing_indicator,
        'night_call_ratio': np.round(night_call_ratio, 4),
        'call_burst_frequency': call_burst_frequency,
        'imei_sim_swap_ratio': np.round(imei_sim_swap_ratio, 4),
        'avg_call_duration_sec': np.round(avg_call_duration_sec, 2),
        'multiple_calling_numbers_indicator': multiple_calling_numbers_indicator,
        'it_act_66d': it_act_66d,
        'telegraph_act_sections': telegraph_act_sections,
        
        'internet_banking_software_targeted': internet_banking_software_targeted,
        'banking_password_compromised': banking_password_compromised,
        'unauthorized_transfer_amount_inr': np.round(unauthorized_transfer_amount_inr, 2),
        'amount_credited_to_accused_ratio': np.round(amount_credited_to_accused_ratio, 4),
        'account_frozen_indicator': account_frozen_indicator,
        'suspicious_credit_not_reported': suspicious_credit_not_reported,
        'bank_correction_request_absent': bank_correction_request_absent,
        'is_fund_transit_velocity_applicable': is_fund_transit_velocity_applicable,
        'fund_transit_velocity_minutes': np.round(fund_transit_velocity_minutes, 2),
        'fan_in_count': fan_in_count,
        'fraudulent_website_count': fraudulent_website_count,
        'payment_gateway_count': payment_gateway_count,
        'linked_beneficiary_account_count': linked_beneficiary_account_count,
        'card_credential_harvesting_indicator': card_credential_harvesting_indicator,
        'it_act_66': it_act_66,
        'it_act_66c': it_act_66c,
        
        'vehicle_assisted_theft': vehicle_assisted_theft,
        'helmet_concealment_indicator': helmet_concealment_indicator,
        'victim_walking_alone_indicator': victim_walking_alone_indicator,
        'victim_approached_from_behind': victim_approached_from_behind,
        'physical_force_used': physical_force_used,
        'stolen_physical_asset_count': stolen_physical_asset_count,
        'stolen_asset_recovered': stolen_asset_recovered,
        'victim_identification_success': victim_identification_success,
        'test_identification_parade_validity': test_identification_parade_validity,
        'reasonable_doubt_identity_gap': reasonable_doubt_identity_gap,
        'ipc_356': ipc_356,
        'ipc_392': ipc_392,
        'ipc_411': ipc_411,
        'ipc_448': ipc_448,
        'ipc_380': ipc_380,
        
        # Topological Network Graph Features (Fix 4)
        'network_degree': network_degree,
        'network_betweenness': np.round(network_betweenness, 4),
        'topological_kingpin_score': np.round(topological_kingpin_score, 4),
        
        'label_entity_role': label_entity_role,
        'label_high_risk_cyber_syndicate': label_high_risk_cyber_syndicate,
        'label_public_order_vs_law_and_order': label_public_order_vs_law_and_order,
        'label_bail_or_custody_relief': label_bail_or_custody_relief,
        'label_conviction_outcome': label_conviction_outcome
    })
    
    elapsed = time.time() - start_time
    return df, elapsed

def export_archetype_subsets(master_df):
    os.makedirs(os.path.join("data", "archetypes"), exist_ok=True)
    
    archetype_configs = [
        {'id': 0, 'name': 'physical_theft', 'filename': 'archetype_0_physical_theft.csv'},
        {'id': 1, 'name': 'atm_distraction', 'filename': 'archetype_1_atm_distraction.csv'},
        {'id': 2, 'name': 'vishing_phishing', 'filename': 'archetype_2_vishing_phishing.csv'},
        {'id': 3, 'name': 'telecom_voip', 'filename': 'archetype_3_telecom_voip.csv'},
        {'id': 4, 'name': 'core_banking_mule', 'filename': 'archetype_4_core_banking_mule.csv'},
        {'id': 5, 'name': 'civilian_benchmark', 'filename': 'archetype_5_civilian_benchmark.csv'}
    ]
    
    subsets_info = []
    for cfg in archetype_configs:
        sub_df = master_df[master_df['archetype_id'] == cfg['id']].copy()
        
        active_cols = []
        for col in sub_df.columns:
            if col == 'archetype_id':
                continue
            uniques = sub_df[col].dropna().unique()
            if len(uniques) > 1:
                active_cols.append(col)
            elif len(uniques) == 1:
                val = uniques[0]
                if val not in [0, 0.0, "0"]:
                    active_cols.append(col)
                    
        sub_df_clean = sub_df[active_cols]
        out_path = os.path.join("data", "archetypes", cfg['filename'])
        sub_df_clean.to_csv(out_path, index=False)
        subsets_info.append((cfg['filename'], len(sub_df_clean), len(active_cols)))
        print(f"[+] Exported Archetype {cfg['id']} ({cfg['name']}) -> {out_path} [{len(sub_df_clean)} rows, {len(active_cols)} active columns]")
        
    return subsets_info

if __name__ == "__main__":
    N_ROWS = 12500
    df, elapsed = generate_synthetic_dataset(N=N_ROWS, seed=42)
    
    # Save master datasets
    output_filename = "synthetic_crime_15k.csv"
    os.makedirs("data", exist_ok=True)
    
    paths_to_save = [
        output_filename,
        os.path.join("data", output_filename)
    ]
    
    # Drop internal helper archetype_id for master file export
    master_df_export = df.drop(columns=['archetype_id'])
    for path in paths_to_save:
        master_df_export.to_csv(path, index=False)
        print(f"[+] Successfully exported master dataset to: {os.path.abspath(path)}")
        
    # Export 6 archetype-specific datasets
    print("\n[+] Generating Archetype-Specific CSV Subsets...")
    subsets_info = export_archetype_subsets(df)
        
    print("\n" + "="*70)
    print("      NCRB SYNTHETIC DATASET REMEDIATION & VALIDATION REPORT")
    print("="*70)
    print(f"Total Generation Time : {elapsed:.3f} seconds (Vectorized Performance)")
    print(f"Total Rows (N)        : {df.shape[0]:,}")
    print(f"Master Feature Count  : {master_df_export.shape[1]}")
    print(f"Missing Values (NaN)  : {master_df_export['fund_transit_velocity_minutes'].isnull().sum():,} in fund_transit_velocity_minutes")
    print("-" * 70)
    print("FIX 1 & CLASS BALANCE: label_entity_role")
    role_names = {
        0: "0: Unwitting / Civilian / Falsely Implicated",
        1: "1: Mastermind / Ghost Coordinator",
        2: "2: Physical Foot Soldier / Street Offender",
        3: "3: Technical Operator / Web Developer / SIM Boxer",
        4: "4: Mule / Financial Beneficiary",
        5: "5: Social Engineer / Call Center Caller"
    }
    counts = df['label_entity_role'].value_counts().sort_index()
    for role_id, count in counts.items():
        pct = (count / len(df)) * 100
        print(f"  {role_names.get(role_id, str(role_id)):<50} : {count:>5} ({pct:>5.2f}%)")
    print("-" * 70)
    print("FIX 4 TOPOLOGICAL KINGPIN vs. MULE METRICS")
    kingpin_df = df[df['label_entity_role'] == 1]
    mule_df = df[df['label_entity_role'] == 4]
    print(f"  Ghost Kingpins (Role 1) -> Avg Degree: {kingpin_df['network_degree'].mean():.2f} | Avg Betweenness: {kingpin_df['network_betweenness'].mean():.4f} | Avg KingpinScore: {kingpin_df['topological_kingpin_score'].mean():.4f}")
    print(f"  Money Mules    (Role 4) -> Avg Degree: {mule_df['network_degree'].mean():.2f} | Avg Betweenness: {mule_df['network_betweenness'].mean():.4f} | Avg KingpinScore: {mule_df['topological_kingpin_score'].mean():.4f}")
    print("="*70)
