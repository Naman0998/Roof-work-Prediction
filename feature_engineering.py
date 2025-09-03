from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features]


class FastFeatureEnricher(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def normalize_roof_type(self, val):
        if pd.isnull(val):
            return 'Unknown'
        val = val.lower().strip()
        if 'shingle' in val:
            return 'Shingle'
        if re.search(r'asphalt|arch', val):
            return 'Shingle'
        if 'tile' in val:
            return 'Tile'
        if 'metal' in val:
            return 'Metal'
        if 'flat' in val:
            return 'Flat'
        if any(x in val for x in ['unknown', 'other', 'n/a', 'na']):
            return 'Unknown'
        return 'Other'

    def transform(self, X):
        df = X.copy()
        id_columns = ['Opportunity ID', 'Opportunity Name', 'Intake Number']
        df = df.drop(columns=[col for col in id_columns if col in df.columns])

        if 'Created Date' in df.columns:
            df['Created Date'] = pd.to_datetime(df['Created Date'], errors='coerce')
            df['intake_month'] = df['Created Date'].dt.month.fillna(6).astype(int)
            df['intake_weekday'] = df['Created Date'].dt.weekday.fillna(3).astype(int)
            df['intake_quarter'] = df['Created Date'].dt.quarter.fillna(2).astype(int)
            df['is_storm_season'] = df['intake_month'].isin([5, 6, 7, 8, 9]).astype(int)
        else:
            df['intake_month'] = 6
            df['intake_weekday'] = 3
            df['intake_quarter'] = 2
            df['is_storm_season'] = 0

        age_map = {
            "0-5 years": 2.5,
            "6-10 years": 8.0,
            "11-15 years": 13.0,
            "16-20 years": 18.0,
            "Above 20 years": 25.0,
            "Unknown": 11.0
        }

        roof_age = df.get('How old is the roof?', pd.Series('Unknown', index=df.index))
        df['roof_age_num'] = roof_age.map(age_map).fillna(20.0)
        df['age_is_unknown'] = (roof_age == 'Unknown').astype(int)

        layers_map = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        roof_layers = df.get('Number of Roof Layers', pd.Series('Unknown', index=df.index)).astype(str)
        df['roof_layers_num'] = roof_layers.map(layers_map)
        df['layers_is_unknown'] = (roof_layers == 'Unknown').astype(int)
        df['roof_layers_num'] = df['roof_layers_num'].fillna(2)

        roof_type = df.get('Type of Roofing', pd.Series('Unknown', index=df.index)).astype(str)
        df['roof_type_norm'] = roof_type.apply(self.normalize_roof_type)

        df['is_old_roof'] = (df['roof_age_num'] >= 15).astype(int)
        df['is_very_old_roof'] = (df['roof_age_num'] >= 20).astype(int)
        df['has_multiple_layers'] = (df['roof_layers_num'] > 1).astype(int)
        df['roof_risk_score'] = df['roof_age_num'] * df['roof_layers_num']

        high_risk_counties = [
            'Queens', 'Fairfield', 'Westchester', 'Staten Island', 'Hartford',
            'New Haven', 'Middlesex', 'Brooklyn', 'Suffolk', 'Nassau'
        ]

        if 'County' in df.columns:
            df['high_risk_county'] = df['County'].isin(high_risk_counties).astype(int)
        else:
            df['high_risk_county'] = 0

        roof_types = ['Shingle', 'Metal', 'Flat', 'Tile', 'Other']
        for rt in roof_types:
            colname = f'roof_type_is_{rt}'
            df[colname] = (df['roof_type_norm'] == rt).astype(int)
            df[f'roof_risk_score_x_{rt}'] = df['roof_risk_score'] * df[colname]

        features = [
            'roof_age_num', 'age_is_unknown', 'roof_layers_num', 'layers_is_unknown',
            'is_old_roof', 'is_very_old_roof', 'has_multiple_layers', 'roof_risk_score',
            'high_risk_county', 'intake_month', 'intake_weekday', 'intake_quarter',
            'is_storm_season',
            'roof_type_is_Shingle', 'roof_type_is_Metal', 'roof_type_is_Flat',
            'roof_type_is_Tile', 'roof_type_is_Other',
            'roof_risk_score_x_Shingle', 'roof_risk_score_x_Metal',
            'roof_risk_score_x_Flat', 'roof_risk_score_x_Tile', 'roof_risk_score_x_Other',
        ]

        return df[features]
