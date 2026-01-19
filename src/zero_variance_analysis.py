"""
Zero Variance Features Analysis
Part 1 of Mobile Biometrics ML Project

Bu script:
1. Tüm kullanıcıların verilerini yükler
2. Her sütunun variance değerini hesaplar
3. Zero variance ve near-zero variance özellikleri tespit eder
4. Sonuçları raporlar
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_sample_sessions(dataset_path: str, num_users: int = 5, num_sessions: int = 3) -> dict:
    """
    Analiz için örnek session'ları yükler.
    
    Args:
        dataset_path: raw-dataset klasörünün yolu
        num_users: Yüklenecek kullanıcı sayısı
        num_sessions: Her kullanıcıdan yüklenecek session sayısı
    
    Returns:
        dict: {
            'touch_data': DataFrame,
            'sensor_data': DataFrame,
            'device_info': DataFrame
        }
    """
    dataset_path = Path(dataset_path)
    
    touch_dfs = []
    sensor_dfs = []
    device_dfs = []
    
    # Kullanıcı klasörlerini al (AA, AB, AC, ...)
    user_dirs = sorted([d for d in dataset_path.iterdir() 
                        if d.is_dir() and d.name != '.DS_Store'])[:num_users]
    
    for user_dir in tqdm(user_dirs, desc="Loading users"):
        pin_dir = user_dir / "Pin"
        if not pin_dir.exists():
            continue
            
        # Session klasörlerini al
        session_dirs = sorted([d for d in pin_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('session_')])[:num_sessions]
        
        for session_dir in session_dirs:
            # Touch events
            touch_file = session_dir / "touch_events.csv"
            if touch_file.exists():
                df = pd.read_csv(touch_file, sep='\t')
                df['user'] = user_dir.name
                df['session'] = session_dir.name
                touch_dfs.append(df)
            
            # Sensor data
            sensor_file = session_dir / "sensor_data.csv"
            if sensor_file.exists():
                df = pd.read_csv(sensor_file, sep='\t')
                df['user'] = user_dir.name
                df['session'] = session_dir.name
                sensor_dfs.append(df)
            
            # Device info
            device_file = session_dir / "device_info.csv"
            if device_file.exists():
                df = pd.read_csv(device_file, sep='\t')
                df['user'] = user_dir.name
                df['session'] = session_dir.name
                device_dfs.append(df)
    
    return {
        'touch_data': pd.concat(touch_dfs, ignore_index=True) if touch_dfs else pd.DataFrame(),
        'sensor_data': pd.concat(sensor_dfs, ignore_index=True) if sensor_dfs else pd.DataFrame(),
        'device_info': pd.concat(device_dfs, ignore_index=True) if device_dfs else pd.DataFrame()
    }


def analyze_variance(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    DataFrame'deki her sütunun variance analizini yapar.
    
    Args:
        df: Analiz edilecek DataFrame
        name: Veri seti adı
    
    Returns:
        DataFrame: Variance analiz sonuçları
    """
    results = []
    
    # Sadece numeric sütunları al
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        variance = df[col].var()
        unique_count = df[col].nunique()
        total_count = len(df[col].dropna())
        most_common = df[col].value_counts().iloc[0] if len(df[col].dropna()) > 0 else 0
        most_common_ratio = most_common / total_count if total_count > 0 else 0
        
        # Zero variance kontrolü
        is_zero_variance = variance == 0 or unique_count == 1
        
        # Near-zero variance kontrolü (caret paketindeki tanım)
        # 1) Unique/Total oranı %10'dan az
        # 2) En yaygın değer ikinci en yaygına göre 19 kat+ fazla
        unique_ratio = unique_count / total_count if total_count > 0 else 0
        freq_ratio = 0
        if unique_count > 1:
            top2 = df[col].value_counts().head(2)
            if len(top2) == 2 and top2.iloc[1] > 0:
                freq_ratio = top2.iloc[0] / top2.iloc[1]
        
        is_near_zero = (unique_ratio < 0.1) or (freq_ratio > 19)
        
        results.append({
            'dataset': name,
            'column': col,
            'variance': variance,
            'unique_count': unique_count,
            'total_count': total_count,
            'unique_ratio': unique_ratio,
            'most_common_ratio': most_common_ratio,
            'freq_ratio': freq_ratio,
            'is_zero_variance': is_zero_variance,
            'is_near_zero_variance': is_near_zero and not is_zero_variance
        })
    
    return pd.DataFrame(results)


def generate_report(results_df: pd.DataFrame, output_path: str):
    """
    Analiz sonuçlarını markdown rapor olarak kaydeder.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Part 1: Zero Variance Features Analysis Report\n\n")
        f.write("Bu rapor, dataset'teki tüm özelliklerin variance analizini içerir.\n\n")
        
        # Özet
        f.write("## Özet\n\n")
        zero_var = results_df[results_df['is_zero_variance']]
        near_zero = results_df[results_df['is_near_zero_variance']]
        
        f.write(f"- **Toplam özellik sayısı:** {len(results_df)}\n")
        f.write(f"- **Zero variance özellikler:** {len(zero_var)}\n")
        f.write(f"- **Near-zero variance özellikler:** {len(near_zero)}\n\n")
        
        # Zero variance detay
        if len(zero_var) > 0:
            f.write("## Zero Variance Özellikler\n\n")
            f.write("Bu özellikler hiçbir ayırt edici bilgi taşımaz ve çıkarılmalıdır.\n\n")
            f.write("| Dataset | Sütun | Unique Count | Variance |\n")
            f.write("|---------|-------|--------------|----------|\n")
            for _, row in zero_var.iterrows():
                f.write(f"| {row['dataset']} | {row['column']} | {row['unique_count']} | {row['variance']:.6f} |\n")
            f.write("\n")
        
        # Near-zero variance detay
        if len(near_zero) > 0:
            f.write("## Near-Zero Variance Özellikler\n\n")
            f.write("Bu özellikler düşük varyansa sahip, dikkatli kullanılmalıdır.\n\n")
            f.write("| Dataset | Sütun | Unique Ratio | Freq Ratio | Variance |\n")
            f.write("|---------|-------|--------------|------------|----------|\n")
            for _, row in near_zero.iterrows():
                f.write(f"| {row['dataset']} | {row['column']} | {row['unique_ratio']:.4f} | {row['freq_ratio']:.2f} | {row['variance']:.6f} |\n")
            f.write("\n")
        
        # Handling methods
        f.write("## Zero Variance Handling Yöntemleri\n\n")
        f.write("### 1. Variance Thresholding (Önerilen)\n\n")
        f.write("**Artıları:**\n")
        f.write("- Basit ve hızlı uygulama\n")
        f.write("- sklearn.feature_selection.VarianceThreshold ile kolay implementasyon\n")
        f.write("- Unsupervised - target değişkene ihtiyaç duymaz\n\n")
        f.write("**Eksileri:**\n")
        f.write("- Kategorik değişkenler için encoding gerektirir\n")
        f.write("- Threshold değeri manuel belirlenir\n\n")
        f.write("```python\n")
        f.write("from sklearn.feature_selection import VarianceThreshold\n")
        f.write("selector = VarianceThreshold(threshold=0)  # Zero variance için\n")
        f.write("X_filtered = selector.fit_transform(X)\n")
        f.write("```\n\n")
        
        f.write("### 2. Doğrudan Silme\n\n")
        f.write("**Artıları:**\n")
        f.write("- En basit yöntem\n")
        f.write("- Model karmaşıklığını azaltır\n\n")
        f.write("**Eksileri:**\n")
        f.write("- Potansiyel bilgi kaybı\n")
        f.write("- Geri dönüşü yok\n\n")
        
        f.write("### 3. Tree-based Modellere Bırakma\n\n")
        f.write("**Artıları:**\n")
        f.write("- Random Forest, XGBoost gibi modeller otomatik ignore eder\n")
        f.write("- Ekstra preprocessing gerektirmez\n\n")
        f.write("**Eksileri:**\n")
        f.write("- Tüm modeller için geçerli değil (SVM, k-NN etkilenir)\n")
        f.write("- Gereksiz hesaplama maliyeti\n\n")
        
        f.write("### 4. Feature-engine DropConstantFeatures\n\n")
        f.write("**Artıları:**\n")
        f.write("- Hem constant hem quasi-constant features için\n")
        f.write("- Kategorik değişkenlerle çalışır\n\n")
        f.write("**Eksileri:**\n")
        f.write("- Ek kütüphane gerektirir\n\n")
        
        f.write("---\n\n")
        f.write("## Sonuç ve Öneriler\n\n")
        if len(zero_var) > 0:
            f.write("1. Yukarıda listelenen **zero variance** özellikler preprocessing aşamasında çıkarılmalıdır.\n")
        if len(near_zero) > 0:
            f.write("2. **Near-zero variance** özellikler için hedef değişkenle korelasyon kontrolü yapılmalıdır.\n")
        f.write("3. Bu projede **VarianceThreshold** kullanılması önerilir.\n")
    
    print(f"Rapor kaydedildi: {output_path}")


def main():
    # Proje kök dizini
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "raw-dataset"
    
    print("=" * 60)
    print("Zero Variance Features Analysis")
    print("=" * 60)
    
    # Verileri yükle (örnek olarak 10 kullanıcı, 5 session)
    print("\n1. Veriler yükleniyor...")
    data = load_sample_sessions(str(dataset_path), num_users=10, num_sessions=5)
    
    print(f"\nYüklenen veriler:")
    print(f"  - Touch data: {len(data['touch_data'])} satır")
    print(f"  - Sensor data: {len(data['sensor_data'])} satır")
    print(f"  - Device info: {len(data['device_info'])} satır")
    
    # Variance analizi
    print("\n2. Variance analizi yapılıyor...")
    results = []
    
    if len(data['touch_data']) > 0:
        results.append(analyze_variance(data['touch_data'], 'touch_events'))
    
    if len(data['sensor_data']) > 0:
        results.append(analyze_variance(data['sensor_data'], 'sensor_data'))
    
    if len(data['device_info']) > 0:
        results.append(analyze_variance(data['device_info'], 'device_info'))
    
    all_results = pd.concat(results, ignore_index=True)
    
    # Sonuçları göster
    print("\n3. Sonuçlar:")
    print("-" * 40)
    
    zero_var = all_results[all_results['is_zero_variance']]
    near_zero = all_results[all_results['is_near_zero_variance']]
    
    print(f"\nZero Variance Özellikler ({len(zero_var)} adet):")
    if len(zero_var) > 0:
        for _, row in zero_var.iterrows():
            print(f"  - [{row['dataset']}] {row['column']}")
    else:
        print("  Yok")
    
    print(f"\nNear-Zero Variance Özellikler ({len(near_zero)} adet):")
    if len(near_zero) > 0:
        for _, row in near_zero.iterrows():
            print(f"  - [{row['dataset']}] {row['column']} (unique_ratio: {row['unique_ratio']:.4f})")
    else:
        print("  Yok")
    
    # Rapor oluştur
    print("\n4. Rapor oluşturuluyor...")
    report_path = project_root / "docs" / "part1_zero_variance.md"
    generate_report(all_results, str(report_path))
    
    # CSV olarak da kaydet
    csv_path = project_root / "output" / "reports" / "variance_analysis.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(csv_path, index=False)
    print(f"CSV kaydedildi: {csv_path}")
    
    print("\n" + "=" * 60)
    print("Analiz tamamlandı!")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    main()
