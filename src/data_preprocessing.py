"""
Module: data_preprocessing.py

Handles raw data cleaning and normalization steps:
- Flags abnormal lab values based on sex-specific ranges
- Splits cleaned data into train/test
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def flag_out_of_range(df):
    df = df.copy()

    df['is_hct_normal'] = ((df['SEX'] == 'M') & df['HAEMATOCRIT'].between(40.0, 52.0)) | \
                          ((df['SEX'] == 'F') & df['HAEMATOCRIT'].between(37.0, 47.0))

    df['is_hb_normal'] = ((df['SEX'] == 'M') & df['HAEMOGLOBINS'].between(13.0, 17.0)) | \
                         ((df['SEX'] == 'F') & df['HAEMOGLOBINS'].between(12.0, 16.0))

    df['is_rbc_normal'] = ((df['SEX'] == 'M') & df['ERYTHROCYTE'].between(4.5, 6.1)) | \
                          ((df['SEX'] == 'F') & df['ERYTHROCYTE'].between(4.0, 5.4))

    df['is_wbc_normal'] = df['LEUCOCYTE'].between(4.0, 10.8)
    df['is_plt_normal'] = df['THROMBOCYTE'].between(150, 400)
    df['is_mch_normal'] = df['MCH'].between(27.0, 33.0)
    df['is_mchc_normal'] = df['MCHC'].between(31.5, 37.0)
    df['is_mcv_normal'] = df['MCV'].between(80, 98)

    return df

def preprocess_data(path: str):
    df = pd.read_csv(path, encoding_errors="ignore")
    df = flag_out_of_range(df)

    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # print("Test columns:", test_data.columns.tolist())

    return train_data, test_data
