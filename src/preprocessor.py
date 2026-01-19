"""
Preprocessor Module
Part 4 of Mobile Biometrics ML Project

Veri ön işleme: Missing value handling, feature scaling, train/test split.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split


class Preprocessor:
    """Veri ön işleme sınıfı."""
    
    # Zero variance olduğu tespit edilen sütunlar (Part 1)
    ZERO_VARIANCE_COLS = ['orientation', 'pressure', 'targetInput']
    
    # Kategorik/ID sütunları (scaling yapılmayacak)
    ID_COLS = ['user', 'session']  # button sadece keystroke-level'da var
    
    def __init__(self, 
                 scaling_method: str = 'standard',
                 variance_threshold: float = 0.0,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Args:
            scaling_method: 'standard' veya 'minmax'
            variance_threshold: Variance threshold değeri
            test_size: Test set oranı
            random_state: Random seed
        """
        self.scaling_method = scaling_method
        self.variance_threshold = variance_threshold
        self.test_size = test_size
        self.random_state = random_state
        
        self.scaler = None
        self.variance_selector = None
        self.feature_columns = None
        self.removed_columns = []
    
    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Numeric sütunları döndürür (ID sütunları hariç)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in self.ID_COLS]
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Missing value'ları işler.
        
        Stratejiler:
        - Numeric: median ile doldur
        - Kategorik: mode ile doldur
        """
        df = df.copy()
        
        numeric_cols = self._get_numeric_columns(df)
        
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        return df
    
    def remove_zero_variance(self, df: pd.DataFrame, 
                             fit: bool = True) -> pd.DataFrame:
        """
        Zero variance sütunları kaldırır.
        """
        df = df.copy()
        numeric_cols = self._get_numeric_columns(df)
        
        # ID sütunlarını ayır
        id_data = df[self.ID_COLS] if any(c in df.columns for c in self.ID_COLS) else None
        feature_data = df[[c for c in numeric_cols if c in df.columns]]
        
        if fit:
            self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
            filtered_data = self.variance_selector.fit_transform(feature_data)
            
            # Hangi sütunlar kaldırıldı?
            mask = self.variance_selector.get_support()
            self.feature_columns = [c for c, m in zip(feature_data.columns, mask) if m]
            self.removed_columns = [c for c, m in zip(feature_data.columns, mask) if not m]
        else:
            filtered_data = self.variance_selector.transform(feature_data)
        
        # DataFrame'e çevir
        result = pd.DataFrame(filtered_data, columns=self.feature_columns, index=df.index)
        
        # ID sütunlarını geri ekle
        if id_data is not None:
            for col in self.ID_COLS:
                if col in id_data.columns:
                    result[col] = id_data[col].values
        
        return result
    
    def scale_features(self, df: pd.DataFrame, 
                       fit: bool = True) -> pd.DataFrame:
        """
        Feature scaling uygular.
        """
        df = df.copy()
        numeric_cols = [c for c in self.feature_columns if c in df.columns]
        
        if fit:
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def split_by_user(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Kullanıcı bazlı train/test split.
        
        Her kullanıcıdan bazı session'lar train, bazıları test'e gider.
        Bu, gerçek senaryoyu simüle eder.
        """
        train_dfs = []
        test_dfs = []
        
        for user in df['user'].unique():
            user_data = df[df['user'] == user]
            sessions = user_data['session'].unique()
            
            train_sessions, test_sessions = train_test_split(
                sessions,
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            train_dfs.append(user_data[user_data['session'].isin(train_sessions)])
            test_dfs.append(user_data[user_data['session'].isin(test_sessions)])
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        return train_df, test_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm preprocessing adımlarını uygular (fit + transform).
        """
        df = self.handle_missing_values(df)
        df = self.remove_zero_variance(df, fit=True)
        df = self.scale_features(df, fit=True)
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Önceden fit edilmiş preprocessor ile transform.
        """
        df = self.handle_missing_values(df)
        df = self.remove_zero_variance(df, fit=False)
        df = self.scale_features(df, fit=False)
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Veriyi model eğitimi için hazırlar.
        
        Returns:
            {
                'X_train': features, 
                'X_test': features,
                'y_train': labels,
                'y_test': labels
            }
        """
        # Train/test split
        train_df, test_df = self.split_by_user(df)
        
        # Preprocessing
        train_processed = self.fit_transform(train_df)
        test_processed = self.transform(test_df)
        
        # Feature/label ayır
        X_train = train_processed[self.feature_columns]
        X_test = test_processed[self.feature_columns]
        
        y_train = train_processed['user']
        y_test = test_processed['user']
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_df': train_processed,
            'test_df': test_processed
        }


def main():
    """Test fonksiyonu."""
    project_root = Path(__file__).parent.parent
    features_path = project_root / "output" / "features" / "session_features.csv"
    
    print("Preprocessor Test")
    print("=" * 60)
    
    # Veri yükle
    df = pd.read_csv(features_path)
    print(f"Loaded: {df.shape}")
    print(f"Users: {df['user'].nunique()}")
    print(f"Sessions: {df['session'].nunique()}")
    
    # Preprocessor
    preprocessor = Preprocessor(
        scaling_method='standard',
        variance_threshold=0.01,  # Near-zero variance da kaldır
        test_size=0.2,
        random_state=42
    )
    
    # Veri hazırla
    data = preprocessor.prepare_data(df)
    
    print(f"\nTrain/Test split:")
    print(f"  X_train: {data['X_train'].shape}")
    print(f"  X_test: {data['X_test'].shape}")
    print(f"  y_train unique: {data['y_train'].nunique()}")
    print(f"  y_test unique: {data['y_test'].nunique()}")
    
    print(f"\nKaldırılan sütunlar ({len(preprocessor.removed_columns)}):")
    print(f"  {preprocessor.removed_columns[:10]}...")
    
    print(f"\nKalan feature sayısı: {len(preprocessor.feature_columns)}")
    
    print("\n" + "=" * 60)
    print("Preprocessor test tamamlandı!")
    
    return data


if __name__ == "__main__":
    main()
