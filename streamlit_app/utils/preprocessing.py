"""
Preprocessing utilities — mirrors the notebook pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_tenant_id(df: pd.DataFrame) -> pd.DataFrame:
    """Detect or derive tenant_id column."""
    df = df.copy()
    tenant_candidates = ['tenant_id', 'entreprise_id', 'entreprise', 'societe', 'societe_nom', 'company']
    tenant_col = next((c for c in tenant_candidates if c in df.columns), None)
    if tenant_col is None:
        if 'client_id' in df.columns:
            tenant_col = 'client_id'
        else:
            raise ValueError(
                "Aucun identifiant tenant trouvé. "
                "Ajoutez `tenant_id`, `entreprise_id`, `entreprise` ou `client_id`."
            )
    df['tenant_id'] = df[tenant_col].astype(str).fillna('unknown')
    return df, tenant_col


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add date & montant features."""
    df = df.copy()
    date_cols = [c for c in df.columns if 'date' in c.lower() and 'facture' in c.lower()]
    date_col = date_cols[0] if date_cols else None
    if date_col:
        df['facture_date'] = pd.to_datetime(df[date_col], errors='coerce')
        df['facture_mois'] = df['facture_date'].dt.month.astype('Int64')
        df['facture_trimestre'] = df['facture_date'].dt.quarter.astype('Int64')
        df['facture_annee'] = df['facture_date'].dt.year.astype('Int64')

    for col in ['regime_tva', 'secteur_activite', 'region', 'ville', 'mode_paiement']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str)

    if 'montant_ttc' in df.columns:
        montant_bins = [0, 100, 500, 2000, 10000, np.inf]
        montant_labels = ['micro', 'petit', 'moyen', 'grand', 'premium']
        df['montant_bucket'] = pd.cut(
            df['montant_ttc'], bins=montant_bins, labels=montant_labels
        ).astype(str)
    return df, date_col


def build_feature_matrix(df: pd.DataFrame):
    """Drop leakage columns and return X, y."""
    leakage = [
        'invoice_id', 'client_id', 'client_nom', 'entreprise', 'product_id',
        'date_facture', 'date_echeance', 'statut_facture', 'retard_paiement',
        'categorie_produit', 'description', 'description_enhanced', 'tenant_id',
        'facture_date'
    ]
    leakage = [c for c in leakage if c in df.columns]
    y = df['retard_paiement'].astype(int).copy() if 'retard_paiement' in df.columns \
        else pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    X = df.drop(columns=leakage).copy()
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numerical_features),
        ('cat', cat_pipe, categorical_features)
    ])
    return preprocessor, numerical_features, categorical_features
