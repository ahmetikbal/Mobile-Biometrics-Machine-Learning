# Mobile Biometrics Authentication using Machine Learning

A behavioral biometric authentication system using PIN entry patterns on mobile devices. This project analyzes touch dynamics and motion sensors to identify users based on their unique interaction patterns.

## 🎯 Results

| Model | Accuracy | F1-Score | EER | FAR | FRR |
|-------|----------|----------|-----|-----|-----|
| **SVM** | **98.33%** | 98.36% | **0.02%** | 0.05% | 0.00% |
| **Random Forest** | **98.33%** | 98.31% | 0.06% | 0.11% | 0.00% |
| k-NN | 97.00% | 96.97% | 0.37% | 0.08% | 0.67% |

> **Best Model:** SVM with RBF kernel (C=10) achieves the lowest Equal Error Rate (0.02%)

## 📊 Dataset

- **30 users** with **60 PIN entry sessions** each
- First 10 sessions excluded as warm-up period
- **1,500 sessions** used for training/testing
- **48,558 keystroke events** extracted

### Data Sources
- Touch events (coordinates, pressure, timing)
- Accelerometer readings
- Gyroscope readings  
- Magnetometer readings

## 🔬 Features Extracted

### Touch-based Features
| Feature | Description |
|---------|-------------|
| Dwell Time | Key hold duration (press → release) |
| Flight Time | Time between consecutive key releases and presses |
| Inter-key Latency | Time between consecutive key presses |
| Position (x, y) | Touch coordinates and drift |

### Sensor-based Features
| Feature | Description |
|---------|-------------|
| Mean, Std, Min, Max | Statistical measures per axis |
| Magnitude | √(x² + y² + z²) for orientation-invariant features |
| Time Window | 50ms before and after each touch event |

## 🏗️ Project Structure

```
├── src/
│   ├── data_loader.py       # Load user sessions from dataset
│   ├── data_merger.py       # Merge touch events with sensor data
│   ├── feature_extractor.py # Extract timing and sensor features
│   ├── preprocessor.py      # Scaling, variance filtering, train/test split
│   └── model_trainer.py     # Training with hyperparameter tuning
├── docs/
│   ├── part1_zero_variance.md   # Zero variance analysis report
│   ├── part2_literature_review.md # Feature derivation references
│   └── ml_tutorial.md           # ML concepts tutorial
├── output/
│   ├── features/            # Extracted feature CSVs
│   ├── models/              # Trained model pickles
│   └── reports/             # Evaluation results
└── raw-dataset/             # Original dataset (not included)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Mobile-Biometrics-Machine-Learning.git
cd Mobile-Biometrics-Machine-Learning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# 1. Extract features from dataset
python src/feature_extractor.py

# 2. Train and evaluate models
python src/model_trainer.py

# 3. Run zero variance analysis
python src/zero_variance_analysis.py
```

## 📈 Methodology

### Preprocessing Pipeline
1. **Missing Value Handling:** Median imputation
2. **Zero Variance Removal:** VarianceThreshold (0.01)
3. **Feature Scaling:** StandardScaler (z-score normalization)
4. **Train/Test Split:** 80/20 user-wise stratified split

### Hyperparameter Tuning
GridSearchCV with 5-fold cross-validation:

| Model | Optimal Parameters |
|-------|-------------------|
| SVM | C=10, kernel=rbf, gamma=scale |
| Random Forest | n_estimators=200, max_depth=None |
| k-NN | n_neighbors=3, weights=distance, metric=manhattan |

### Evaluation Metrics
- **Accuracy, Precision, Recall, F1-Score**
- **EER (Equal Error Rate):** Point where FAR = FRR
- **FAR (False Acceptance Rate):** Impostor accepted
- **FRR (False Rejection Rate):** Genuine user rejected

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Zero Variance Analysis](docs/part1_zero_variance.md) | Constant feature detection |
| [Literature Review](docs/part2_literature_review.md) | Academic references for features |
| [ML Tutorial](docs/ml_tutorial.md) | Machine learning concepts guide |
| [Results Table](output/reports/results_table.md) | Detailed evaluation metrics |

## 🔧 Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- tqdm >= 4.65.0

## 📝 Key Findings

1. **Pressure data is unusable:** All devices returned constant values (zero variance)
2. **Touch area (major/minor):** Near-zero variance, requires careful handling
3. **SVM outperforms** other models with lowest EER (0.02%)
4. **50ms time window** is sufficient for sensor-touch alignment
5. **Hyperparameter tuning** improved SVM accuracy from 94.5% → 98.33%

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Dataset collected from mobile PIN entry sessions
- Inspired by keystroke dynamics and touchalytics research

---

**Last Updated:** January 2026
