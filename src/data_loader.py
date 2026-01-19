"""
Data Loader Module
Part 3 of Mobile Biometrics ML Project

Tüm kullanıcıların verilerini yükler ve organize eder.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class DataLoader:
    """Dataset yükleyici sınıfı."""
    
    # Kullanılacak sensör tipleri (PDF'te belirtilen)
    REQUIRED_SENSORS = ['ACCELEROMETER', 'GYROSCOPE', 'MAGNETOMETER']
    
    # Warm-up session sayısı (ilk 10 session)
    WARMUP_SESSIONS = 10
    
    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: raw-dataset klasörünün yolu
        """
        self.dataset_path = Path(dataset_path)
        self.users = self._get_users()
        
    def _get_users(self) -> List[str]:
        """Kullanıcı listesini döndürür."""
        user_dirs = sorted([
            d.name for d in self.dataset_path.iterdir()
            if d.is_dir() and d.name != '.DS_Store'
        ])
        return user_dirs
    
    def _get_sessions(self, user: str, exclude_warmup: bool = True) -> List[str]:
        """
        Kullanıcının session listesini döndürür.
        
        Args:
            user: Kullanıcı ID
            exclude_warmup: True ise ilk 10 session'ı çıkarır
        """
        pin_dir = self.dataset_path / user / "Pin"
        if not pin_dir.exists():
            return []
        
        sessions = sorted([
            d.name for d in pin_dir.iterdir()
            if d.is_dir() and d.name.startswith('session_')
        ], key=lambda x: int(x.split('_')[1]))
        
        if exclude_warmup:
            sessions = sessions[self.WARMUP_SESSIONS:]
        
        return sessions
    
    def load_session(self, user: str, session: str) -> Dict[str, pd.DataFrame]:
        """
        Tek bir session'ın verilerini yükler.
        
        Returns:
            {
                'touch': DataFrame,
                'sensor': DataFrame,
                'device': DataFrame
            }
        """
        session_dir = self.dataset_path / user / "Pin" / session
        
        result = {}
        
        # Touch events
        touch_file = session_dir / "touch_events.csv"
        if touch_file.exists():
            df = pd.read_csv(touch_file, sep='\t')
            df['user'] = user
            df['session'] = session
            result['touch'] = df
        
        # Sensor data
        sensor_file = session_dir / "sensor_data.csv"
        if sensor_file.exists():
            df = pd.read_csv(sensor_file, sep='\t')
            # Sadece gerekli sensörleri filtrele
            df = df[df['sensorType'].isin(self.REQUIRED_SENSORS)]
            df['user'] = user
            df['session'] = session
            result['sensor'] = df
        
        # Device info
        device_file = session_dir / "device_info.csv"
        if device_file.exists():
            df = pd.read_csv(device_file, sep='\t')
            df['user'] = user
            df['session'] = session
            result['device'] = df
        
        return result
    
    def load_user(self, user: str, exclude_warmup: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Bir kullanıcının tüm session'larını yükler.
        """
        sessions = self._get_sessions(user, exclude_warmup)
        
        touch_dfs = []
        sensor_dfs = []
        device_dfs = []
        
        for session in sessions:
            data = self.load_session(user, session)
            if 'touch' in data:
                touch_dfs.append(data['touch'])
            if 'sensor' in data:
                sensor_dfs.append(data['sensor'])
            if 'device' in data:
                device_dfs.append(data['device'])
        
        return {
            'touch': pd.concat(touch_dfs, ignore_index=True) if touch_dfs else pd.DataFrame(),
            'sensor': pd.concat(sensor_dfs, ignore_index=True) if sensor_dfs else pd.DataFrame(),
            'device': pd.concat(device_dfs, ignore_index=True) if device_dfs else pd.DataFrame()
        }
    
    def load_all(self, exclude_warmup: bool = True, 
                 max_users: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Tüm kullanıcıların verilerini yükler.
        
        Args:
            exclude_warmup: Warm-up session'ları çıkar
            max_users: Yüklenecek maksimum kullanıcı sayısı (test için)
        """
        users = self.users[:max_users] if max_users else self.users
        
        touch_dfs = []
        sensor_dfs = []
        device_dfs = []
        
        for user in tqdm(users, desc="Loading users"):
            data = self.load_user(user, exclude_warmup)
            if len(data['touch']) > 0:
                touch_dfs.append(data['touch'])
            if len(data['sensor']) > 0:
                sensor_dfs.append(data['sensor'])
            if len(data['device']) > 0:
                device_dfs.append(data['device'])
        
        return {
            'touch': pd.concat(touch_dfs, ignore_index=True) if touch_dfs else pd.DataFrame(),
            'sensor': pd.concat(sensor_dfs, ignore_index=True) if sensor_dfs else pd.DataFrame(),
            'device': pd.concat(device_dfs, ignore_index=True) if device_dfs else pd.DataFrame()
        }
    
    def get_keystroke_events(self, touch_df: pd.DataFrame) -> pd.DataFrame:
        """
        Touch events'lerden keystroke (tuş basım) olaylarını çıkarır.
        
        Her keystroke için başlangıç ve bitiş zamanlarını hesaplar.
        """
        # action=0: touch starts, action=1: touch ends
        # Aynı tuş basımını grupla
        
        keystrokes = []
        
        for (user, session), group in touch_df.groupby(['user', 'session']):
            # Sırala
            group = group.sort_values('timestampMs')
            
            # action=0 ve action=1 eşleştir
            press_events = group[group['action'] == 0].reset_index(drop=True)
            release_events = group[group['action'] == 1].reset_index(drop=True)
            
            # Basit eşleştirme: her press için en yakın sonraki release
            for i, press in press_events.iterrows():
                # Bu press'ten sonraki release'leri bul
                later_releases = release_events[
                    release_events['timestampMs'] > press['timestampMs']
                ]
                
                if len(later_releases) > 0:
                    release = later_releases.iloc[0]
                    
                    keystrokes.append({
                        'user': user,
                        'session': session,
                        'button': press['pressedButton'],
                        'press_time': press['timestampMs'],
                        'release_time': release['timestampMs'],
                        'dwell_time': release['timestampMs'] - press['timestampMs'],
                        'x': press['x'],
                        'y': press['y'],
                        'pressure': press['pressure'],
                        'major': press['major'],
                        'minor': press['minor'],
                        'size': press['size']
                    })
        
        return pd.DataFrame(keystrokes)


def main():
    """Test fonksiyonu."""
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "raw-dataset"
    
    print("DataLoader Test")
    print("=" * 50)
    
    loader = DataLoader(str(dataset_path))
    
    print(f"Toplam kullanıcı: {len(loader.users)}")
    print(f"Kullanıcılar: {loader.users[:5]}...")
    
    # Tek kullanıcı yükle
    print("\n1 kullanıcı yükleniyor (warm-up hariç)...")
    user_data = loader.load_user('AA', exclude_warmup=True)
    
    print(f"  Touch events: {len(user_data['touch'])} satır")
    print(f"  Sensor data: {len(user_data['sensor'])} satır")
    
    # Keystroke events çıkar
    print("\nKeystroke events çıkarılıyor...")
    keystrokes = loader.get_keystroke_events(user_data['touch'])
    print(f"  Keystroke sayısı: {len(keystrokes)}")
    
    if len(keystrokes) > 0:
        print(f"\n  Örnek keystroke:")
        print(keystrokes.head(3).to_string())
        
        print(f"\n  Dwell time stats:")
        print(f"    Mean: {keystrokes['dwell_time'].mean():.2f} ms")
        print(f"    Std: {keystrokes['dwell_time'].std():.2f} ms")
        print(f"    Min: {keystrokes['dwell_time'].min():.2f} ms")
        print(f"    Max: {keystrokes['dwell_time'].max():.2f} ms")
    
    print("\n" + "=" * 50)
    print("Test tamamlandı!")


if __name__ == "__main__":
    main()
