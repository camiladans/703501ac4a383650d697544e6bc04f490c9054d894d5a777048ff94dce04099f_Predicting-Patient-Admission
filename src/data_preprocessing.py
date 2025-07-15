import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

"""
https://pmc.ncbi.nlm.nih.gov/articles/PMC9687310/#sec2-biomedicines-10-02697
Haematocrit (HCT): Male normal range is 40.0–52.0%, while Female normal range is	37.0–47.0%
HAEMOGLOBINS (Hb): Male normal range is	13.0–17.0 g/dL while female normal range is	12.0–16.0 g/dL
ERYTHROCYTE (RBC): Male normal range is	4.5–6.1 × 106/μL, while female normal range is	4.0–5.4 × 106/μL
LEUCOCYTE (WBC): Male and female normal range is	4.0–10.8 × 103/μL
THROMBOCYTE (PLT): Male and female normal range is	150–400 × 103/μL
MCH: Male and female normal range is 27.0–33.0 pg
MCHC: Male and female normal range is	31.5–37.0 g/dL
MCV: Male and female normal range	80–98 fL	80–98 fL
"""

def flag_out_of_range(df):
    df = df.copy()

    # HAEMATOCRIT
    df['is_hct_normal'] = (
        ((df['SEX'] == 'M') & df['HAEMATOCRIT'].between(40.0, 52.0)) |
        ((df['SEX'] == 'F') & df['HAEMATOCRIT'].between(37.0, 47.0))
    )

    # Haemoglobin
    df['is_hb_normal'] = (
        ((df['SEX'] == 'M') & df['HAEMOGLOBINS'].between(13.0, 17.0)) |
        ((df['SEX'] == 'F') & df['HAEMOGLOBINS'].between(12.0, 16.0))
    )

    # RBC or ERYTHROCYTE
    df['is_rbc_normal'] = (
        ((df['SEX'] == 'M') & df['ERYTHROCYTE'].between(4.5, 6.1)) |
        ((df['SEX'] == 'F') & df['ERYTHROCYTE'].between(4.0, 5.4))
    )

    # WBC or LEUCOCYTE
    df['is_wbc_normal'] = df['LEUCOCYTE'].between(4.0, 10.8)

    # THROMBOCYTE or Platelet
    df['is_plt_normal'] = df['THROMBOCYTE'].between(150, 400)

    # MCH
    df['is_mch_normal'] = df['MCH'].between(27.0, 33.0)

    # MCHC
    df['is_mchc_normal'] = df['MCHC'].between(31.5, 37.0)

    # MCV
    df['is_mcv_normal'] = df['MCV'].between(80, 98)

    return df


def load_and_preprocess_data(path):
    # Load Excel data
    df = pd.read_excel(path)

    # Optional: drop rows with missing target
    df = df.dropna(subset=['SOURCE']) #Source indicates whether patient was "IN"-patient or "OUT"-patient

    # Add binary normal/abnormal lab flags
    df = flag_out_of_range(df)

    # Encode SEX column
    df = pd.get_dummies(df, columns=['SEX'], drop_first=True)

    # Separate features and target 
    y = df['SOURCE']  # Replace with your actual target column
    X = df.drop(columns=['SOURCE'])

    # Impute and scale
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    # Train-test split
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
