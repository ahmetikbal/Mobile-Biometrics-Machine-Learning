"""
Feature Extractor Module
Part 3 of Mobile Biometrics ML Project

Touch ve sensor verilerinden derived featureları çıkarır.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from data_loader import DataLoader
from data_merger import DataMerger


class FeatureExtractor:
    """Feature extraction sınıfı."""
    
    # Zero variance olduğu tespit edilen sütunlar
    ZERO_VARIANCE_COLS = ['orientation', 'pressure', 'targetInput']
    
    # Near-zero variance sütunlar (dikkatli kullanılmalı)
    NEAR_ZERO_VARIANCE_COLS = ['major', 'minor', 'size']
    
    def __init__(self, time_window_ms: int = 50):
        self.time_window_ms = time_window_ms
        self.merger = DataMerger(time_window_ms)
    
    def extract_timing_features(self, keystrokes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Keystroke timing özelliklerini çıkarır.
        
        Features:
        - dwell_time: Tuşa basılı kalma süresi
        - flight_time: Tuşlar arası bekleme süresi
        - inter_key_latency: Ardışık basımlar arası süre
        """
        df = keystrokes_df.copy()
        
        # Dwell time zaten var
        # Flight time hesapla
        df['flight_time'] = np.nan
        df['inter_key_latency'] = np.nan
        
        # Her session için ayrı hesapla
        for (user, session), group in df.groupby(['user', 'session']):
            group = group.sort_values('press_time')
            indices = group.index
            
            for i in range(1, len(indices)):
                prev_idx = indices[i-1]
                curr_idx = indices[i]
                
                # Flight time: önceki release - şimdiki press
                flight = df.loc[curr_idx, 'press_time'] - df.loc[prev_idx, 'release_time']
                df.loc[curr_idx, 'flight_time'] = flight
                
                # Inter-key latency: önceki press - şimdiki press
                latency = df.loc[curr_idx, 'press_time'] - df.loc[prev_idx, 'press_time']
                df.loc[curr_idx, 'inter_key_latency'] = latency
        
        return df
    
    def extract_position_features(self, keystrokes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pozisyon bazlı özellikler çıkarır.
        
        Features:
        - x_diff: Ardışık tuşlar arası x farkı
        - y_diff: Ardışık tuşlar arası y farkı
        - distance: Ardışık tuşlar arası mesafe
        """
        df = keystrokes_df.copy()
        
        df['x_diff'] = np.nan
        df['y_diff'] = np.nan
        df['distance'] = np.nan
        
        for (user, session), group in df.groupby(['user', 'session']):
            group = group.sort_values('press_time')
            indices = group.index
            
            for i in range(1, len(indices)):
                prev_idx = indices[i-1]
                curr_idx = indices[i]
                
                x_diff = df.loc[curr_idx, 'x'] - df.loc[prev_idx, 'x']
                y_diff = df.loc[curr_idx, 'y'] - df.loc[prev_idx, 'y']
                distance = np.sqrt(x_diff**2 + y_diff**2)
                
                df.loc[curr_idx, 'x_diff'] = x_diff
                df.loc[curr_idx, 'y_diff'] = y_diff
                df.loc[curr_idx, 'distance'] = distance
        
        return df
    
    def extract_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Session-level aggregate özellikler çıkarır.
        Her session için bir feature vector oluşturur.
        """
        session_features = []
        
        for (user, session), group in df.groupby(['user', 'session']):
            features = {
                'user': user,
                'session': session,
                'keystroke_count': len(group),
            }
            
            # Timing features
            for col in ['dwell_time', 'flight_time', 'inter_key_latency']:
                if col in group.columns:
                    features[f'{col}_mean'] = group[col].mean()
                    features[f'{col}_std'] = group[col].std()
                    features[f'{col}_min'] = group[col].min()
                    features[f'{col}_max'] = group[col].max()
            
            # Position features
            for col in ['x_diff', 'y_diff', 'distance']:
                if col in group.columns:
                    features[f'{col}_mean'] = group[col].mean()
                    features[f'{col}_std'] = group[col].std()
            
            # Sensor features - aggregate
            sensor_cols = [c for c in group.columns if c.startswith(('acc_', 'gyr_', 'mag_'))]
            for col in sensor_cols:
                if col in group.columns:
                    features[f'{col}_session_mean'] = group[col].mean()
            
            session_features.append(features)
        
        return pd.DataFrame(session_features)
    
    def process_user(self, user_data: Dict[str, pd.DataFrame], 
                     loader: DataLoader) -> pd.DataFrame:
        """
        Tek bir kullanıcının tüm verilerini işler.
        """
        # Keystroke events çıkar
        keystrokes = loader.get_keystroke_events(user_data['touch'])
        
        if len(keystrokes) == 0:
            return pd.DataFrame()
        
        # Timing features
        keystrokes = self.extract_timing_features(keystrokes)
        
        # Position features
        keystrokes = self.extract_position_features(keystrokes)
        
        # Sensor features (merge)
        merged = self.merger.merge_keystroke_with_sensors(
            keystrokes, user_data['sensor'], show_progress=False
        )
        
        return merged
    
    def process_all_users(self, dataset_path: str, 
                          max_users: Optional[int] = None,
                          output_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Tüm kullanıcıları işler.
        
        Returns:
            (keystroke_features, session_features)
        """
        loader = DataLoader(dataset_path)
        users = loader.users[:max_users] if max_users else loader.users
        
        all_keystrokes = []
        
        for user in tqdm(users, desc="Processing users"):
            user_data = loader.load_user(user, exclude_warmup=True)
            
            if len(user_data['touch']) == 0:
                continue
            
            merged = self.process_user(user_data, loader)
            
            if len(merged) > 0:
                all_keystrokes.append(merged)
        
        if not all_keystrokes:
            return pd.DataFrame(), pd.DataFrame()
        
        # Birleştir
        keystroke_df = pd.concat(all_keystrokes, ignore_index=True)
        
        # Session-level features
        session_df = self.extract_session_features(keystroke_df)
        
        # Kaydet
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            keystroke_df.to_csv(output_path / 'keystroke_features.csv', index=False)
            session_df.to_csv(output_path / 'session_features.csv', index=False)
            print(f"Saved: {output_path}")
        
        return keystroke_df, session_df


def main():
    """Ana fonksiyon."""
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "raw-dataset"
    output_path = project_root / "output" / "features"
    
    print("Feature Extraction")
    print("=" * 60)
    
    extractor = FeatureExtractor(time_window_ms=50)
    
    # Tüm 30 kullanıcıyı işle
    print("\n30 kullanıcı işleniyor...")
    keystroke_df, session_df = extractor.process_all_users(
        str(dataset_path),
        max_users=30,  # Tüm kullanıcılar
        output_path=str(output_path)
    )
    
    print(f"\nKeystroke features: {keystroke_df.shape}")
    print(f"Session features: {session_df.shape}")
    
    if len(session_df) > 0:
        print(f"\nSession feature sütunları ({len(session_df.columns)} adet):")
        print(list(session_df.columns)[:20], "...")
        
        print(f"\nKullanıcı başına session sayısı:")
        print(session_df['user'].value_counts())
    
    print("\n" + "=" * 60)
    print("Feature extraction tamamlandı!")
    
    return keystroke_df, session_df


if __name__ == "__main__":
    main()
