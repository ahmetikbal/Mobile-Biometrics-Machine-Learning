# Mobile Biometrics Machine Learning

PIN giriş davranışsal biyometrik kimlik doğrulama sistemi.

## Kurulum

```bash
cd /Users/loodos/Desktop/Mobile-Biometrics-Machine-Learning
source .venv/bin/activate
pip install -r requirements.txt
```

## Proje Yapısı

```
├── raw-dataset/       # Ham veri seti (30 kullanıcı × 60 session)
├── src/               # Python kaynak kodları
├── docs/              # Dokümantasyon
├── notebooks/         # Jupyter notebooks
└── output/            # Çıktılar (modeller, raporlar, grafikler)
```

## Kullanım

```bash
# Zero variance analizi
python src/zero_variance_analysis.py

# Feature extraction
python src/feature_extractor.py

# Model eğitimi
python src/model_trainer.py
```
