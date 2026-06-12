import numpy as np
import pandas as pd
import os

def generate_mock_ehr_data(num_samples=1000, output_path="datasets/ehr_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)

    age = np.random.normal(50, 15, num_samples).clip(18, 90)
    sex = np.random.binomial(1, 0.5, num_samples)

    visit_freq = np.random.poisson(3, num_samples)
    icd9_codes_count = np.random.poisson(2, num_samples)
    icd10_codes_count = np.random.poisson(4, num_samples)
    prior_psych = np.random.binomial(1, 0.1, num_samples)

    X = np.column_stack([
        age,
        sex,
        visit_freq,
        icd9_codes_count,
        icd10_codes_count,
        prior_psych
    ])

    logits = -2.5 + 2.0 * prior_psych + 0.1 * visit_freq + 0.02 * age
    probs = 1 / (1 + np.exp(-logits))

    y = np.random.binomial(1, probs)

    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'visit_frequency': visit_freq,
        'icd9_count': icd9_codes_count,
        'icd10_count': icd10_codes_count,
        'prior_psychiatric_diagnosis': prior_psych,
        'depression_diagnosis': y
    })

    mask = np.random.rand(num_samples) < 0.05
    df.loc[mask, 'visit_frequency'] = np.nan

    df.to_csv(output_path, index=False)
    print(f"Mock EHR data saved to {output_path} with {num_samples} samples.")

if __name__ == "__main__":
    generate_mock_ehr_data()