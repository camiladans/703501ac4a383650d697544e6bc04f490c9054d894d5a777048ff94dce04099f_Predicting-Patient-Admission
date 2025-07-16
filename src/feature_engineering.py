"""
Module: feature_engineering.py

This module handles feature transformations including:
- Scaling numeric features
- Polynomial feature expansion on numeric data
- Low-variance feature removal
- Encoding categorical features
Preserves 'SOURCE' column if present.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler #PolynomialFeatures
# from sklearn.feature_selection import VarianceThreshold

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes only the numeric features to zero mean and unit variance.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with numeric columns scaled, others unchanged.
    """
    df = df.copy()
    source = df.pop("SOURCE") if "SOURCE" in df.columns else None

    numeric_cols = df.select_dtypes(include="number").columns
    non_numeric_cols = df.columns.difference(numeric_cols)

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[numeric_cols])
    df_scaled = pd.DataFrame(scaled_array, columns=numeric_cols, index=df.index)

    df_out = pd.concat([df_scaled, df[non_numeric_cols]], axis=1)
    df_out = df_out[df.columns]  # preserve original column order

    if source is not None:
        df_out["SOURCE"] = source

    return df_out

# def add_polynomial_features(df: pd.DataFrame, degree=2) -> pd.DataFrame:
#     """
#     Generates polynomial features up to the specified degree on numeric columns only.

#     Args:
#         df (pd.DataFrame): DataFrame that may contain both numeric and non-numeric features.
#         degree (int): Maximum degree of polynomial features to generate.

#     Returns:
#         pd.DataFrame: DataFrame with polynomial features and non-numeric columns preserved.
#     """
#     df = df.copy()
#     source = df.pop("SOURCE") if "SOURCE" in df.columns else None

#     numeric_cols = df.select_dtypes(include="number").columns
#     non_numeric_cols = df.columns.difference(numeric_cols)

#     poly = PolynomialFeatures(degree=degree, include_bias=False)
#     poly_array = poly.fit_transform(df[numeric_cols])
#     poly_df = pd.DataFrame(poly_array, columns=poly.get_feature_names_out(numeric_cols), index=df.index)

#     df_out = pd.concat([poly_df, df[non_numeric_cols]], axis=1)
#     if source is not None:
#         df_out["SOURCE"] = source

#     return df_out

# def remove_low_variance_features(df: pd.DataFrame, threshold=0.01) -> pd.DataFrame:
#     """
#     Removes features with variance below the specified threshold, applied only to numeric columns.

#     Args:
#         df (pd.DataFrame): DataFrame with features.
#         threshold (float): Minimum variance required to retain a feature.

#     Returns:
#         pd.DataFrame: DataFrame with low-variance numeric features removed, non-numeric features preserved.
#     """
#     df = df.copy()
#     source = df.pop("SOURCE") if "SOURCE" in df.columns else None

#     numeric_cols = df.select_dtypes(include="number").columns
#     non_numeric_cols = df.columns.difference(numeric_cols)

#     selector = VarianceThreshold(threshold=threshold)
#     reduced_array = selector.fit_transform(df[numeric_cols])
#     selected_columns = numeric_cols[selector.get_support()]

#     reduced_df = pd.DataFrame(reduced_array, columns=selected_columns, index=df.index)
#     df_out = pd.concat([reduced_df, df[non_numeric_cols]], axis=1)

#     if source is not None:
#         df_out["SOURCE"] = source

#     return df_out

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes 'SEX' column using one-hot encoding. Ignores other categorical columns.

    Args:
        df (pd.DataFrame): DataFrame with numerical and categorical features.

    Returns:
        pd.DataFrame: DataFrame with 'SEX' encoded, all others unchanged.
    """
    df = df.copy()
    if "SEX" in df.columns:
        df = pd.get_dummies(df, columns=["SEX"], drop_first=True)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies full feature engineering pipeline: scaling, polynomial features,
    low-variance feature removal, and categorical encoding.

    Args:
        df (pd.DataFrame): Raw DataFrame with numerical and categorical features.

    Returns:
        pd.DataFrame: Fully numeric engineered feature set.
    """
    df_scaled = scale_features(df)
    # df_poly = add_polynomial_features(df_scaled)
    # df_reduced = remove_low_variance_features(df_poly)
    df_encoded = encode_categorical_features(df_scaled)
    return df_encoded
