"""
Data Merger Module
Part 3 of Mobile Biometrics ML Project

Touch events ve sensor verilerini time window ile birleştirir.
(Optimized v2 - Using numpy for faster processing)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from data_loader import DataLoader


class DataMerger:
    """Touch ve sensor verilerini birleştiren sınıf (Optimized)."""
    
    def __init__(self, time_window_ms: int = 50):
        """
        Args:
            time_window_ms: Touch event etrafındaki sensör penceresi (ms)
        """
        self.time_window_ms = time_window_ms
    
    def _preprocess_sensor_data(self, sensor_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Sensör verilerini tip bazında gruplar ve sıralar (performans için).
        """
        sensor_dict = {}
        for sensor_type in ['ACCELEROMETER', 'GYROSCOPE', 'MAGNETOMETER']:
            df = sensor_df[sensor_df['sensorType'] == sensor_type].copy()
            df = df.sort_values('timestampMs').reset_index(drop=True)
            sensor_dict[sensor_type] = df
        return sensor_dict
    
    def _get_window_stats_fast(self, 
                               sensor_data: pd.DataFrame,
                               timestamp: float) -> Dict[str, float]:
        """
        Hızlı sensör window istatistikleri (searchsorted kullanarak).
        """
        if len(sensor_data) == 0:
            return {key: np.nan for key in [
                'mean_x', 'mean_y', 'mean_z',
                'std_x', 'std_y', 'std_z',
                'min_x', 'min_y', 'min_z',
                'max_x', 'max_y', 'max_z',
                'range_x', 'range_y', 'range_z',
                'magnitude_mean', 'magnitude_std'
            ]}
        
        timestamps = sensor_data['timestampMs'].values
        
        start_time = timestamp - self.time_window_ms
        end_time = timestamp + self.time_window_ms
        
        # Binary search ile indeks bul
        start_idx = np.searchsorted(timestamps, start_time, side='left')
        end_idx = np.searchsorted(timestamps, end_time, side='right')
        
        if start_idx >= end_idx:
            return {key: np.nan for key in [
                'mean_x', 'mean_y', 'mean_z',
                'std_x', 'std_y', 'std_z',
                'min_x', 'min_y', 'min_z',
                'max_x', 'max_y', 'max_z',
                'range_x', 'range_y', 'range_z',
                'magnitude_mean', 'magnitude_std'
            ]}
        
        # Window verilerini al
        x = sensor_data['rawX'].values[start_idx:end_idx]
        y = sensor_data['rawY'].values[start_idx:end_idx]
        z = sensor_data['rawZ'].values[start_idx:end_idx]
        
        # Magnitude
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        
        return {
            'mean_x': np.mean(x),
            'mean_y': np.mean(y),
            'mean_z': np.mean(z),
            'std_x': np.std(x),
            'std_y': np.std(y),
            'std_z': np.std(z),
            'min_x': np.min(x),
            'min_y': np.min(y),
            'min_z': np.min(z),
            'max_x': np.max(x),
            'max_y': np.max(y),
            'max_z': np.max(z),
            'range_x': np.max(x) - np.min(x),
            'range_y': np.max(y) - np.min(y),
            'range_z': np.max(z) - np.min(z),
            'magnitude_mean': np.mean(magnitude),
            'magnitude_std': np.std(magnitude)
        }
    
    def merge_keystroke_with_sensors(self, 
                                     keystrokes_df: pd.DataFrame,
                                     sensor_df: pd.DataFrame,
                                     show_progress: bool = True) -> pd.DataFrame:
        """
        Keystroke verileriyle sensör verilerini birleştirir (Optimized).
        """
        # Sensör verilerini ön-işle
        sensor_dict = self._preprocess_sensor_data(sensor_df)
        
        merged_rows = []
        
        iterator = keystrokes_df.iterrows()
        if show_progress:
            iterator = tqdm(list(iterator), desc="Merging keystrokes")
        
        for idx, keystroke in iterator:
            row = keystroke.to_dict()
            
            # Press anındaki sensörler
            for sensor_type in ['ACCELEROMETER', 'GYROSCOPE', 'MAGNETOMETER']:
                prefix = sensor_type[:3].lower()  # acc, gyr, mag
                
                stats = self._get_window_stats_fast(
                    sensor_dict[sensor_type],
                    keystroke['press_time']
                )
                
                for key, value in stats.items():
                    row[f'{prefix}_{key}'] = value
            
            merged_rows.append(row)
        
        return pd.DataFrame(merged_rows)



def main():
    """Test fonksiyonu."""
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "raw-dataset"
    
    print("DataMerger Test")
    print("=" * 50)
    
    # Veri yükle
    loader = DataLoader(str(dataset_path))
    user_data = loader.load_user('AA', exclude_warmup=True)
    
    print(f"Touch events: {len(user_data['touch'])} satır")
    print(f"Sensor data: {len(user_data['sensor'])} satır")
    
    # Keystroke events çıkar
    keystrokes = loader.get_keystroke_events(user_data['touch'])
    print(f"Keystroke events: {len(keystrokes)} adet")
    
    # Merger oluştur
    merger = DataMerger(time_window_ms=50)
    
    # Overlap kontrolü
    timestamps = keystrokes['press_time'].tolist()[:20]
    overlaps = merger.check_overlap(timestamps)
    overlap_count = sum(overlaps)
    print(f"\nOverlap durumu (ilk 20): {overlap_count}/{len(overlaps)} çakışma")
    
    # Merge işlemi (örnek olarak ilk 100 keystroke)
    print("\nSensör verileriyle birleştiriliyor (ilk 100)...")
    sample_keystrokes = keystrokes.head(100)
    merged = merger.merge_keystroke_with_sensors(
        sample_keystrokes, 
        user_data['sensor'],
        show_progress=True
    )
    
    print(f"\nBirleştirilmiş veri: {merged.shape}")
    print(f"Sütunlar: {list(merged.columns)}")
    
    # NaN kontrolü
    nan_cols = merged.columns[merged.isna().any()].tolist()
    print(f"\nNaN içeren sütunlar: {len(nan_cols)}")
    
    print("\n" + "=" * 50)
    print("Test tamamlandı!")
    
    return merged


if __name__ == "__main__":
    main()
