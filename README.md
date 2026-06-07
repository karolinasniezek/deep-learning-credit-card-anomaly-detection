# Credit Card Anomaly Detection with Autoencoders

Unsupervised anomaly detection pipeline implemented using TensorFlow/Keras autoencoders.

## Technologies

* Python 3.10
* TensorFlow / Keras
* Scikit-Learn
* NumPy
* Pandas
* Matplotlib

## Dataset

Credit card transaction dataset containing:

* Normal transactions
* Fraudulent transactions

Target column:

```text
Class
```

* `0` = Normal Transaction
* `1` = Fraudulent Transaction

## Implemented

### Data Preprocessing

* Feature standardization using `StandardScaler`
* Separation of normal and anomalous transactions
* Training on non-fraudulent transactions only

### Autoencoder Architecture

Encoder:

```text
Input (29)
 ├─ Dense(32, ReLU)
 ├─ BatchNormalization
 ├─ Dropout(0.2)
 └─ Dense(14, ReLU)
```

Decoder:

```text
Dense(32, ReLU)
BatchNormalization
Dropout(0.2)
Dense(29, Linear)
```

### Model Training

* Optimizer: Adam
* Learning Rate: 0.001
* Loss Function: Mean Squared Error (MSE)
* Validation Split: 10%
* Batch Size: 256
* Epochs: 20

### Anomaly Detection

* Transaction reconstruction using autoencoder
* Reconstruction error calculation (MSE)
* 95th percentile thresholding
* Binary anomaly classification

```python
threshold = np.percentile(
    mse[y == 0],
    95
)
```

### Model Evaluation

Implemented metrics:

* ROC-AUC Score
* Classification Report
* ROC Curve

## Visualizations

### ROC Curve

![ROC Curve](figures/roc_curve.png)

The ROC curve illustrates the trade-off between True Positive Rate and False Positive Rate across different anomaly score thresholds.

## Project Structure

```text
.
├── data/
│   └── creditcard.csv
│
├── figures/
│   └── roc_curve.png
│
├── src/
│   └── anomaly_detection.py
│
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/karolinasniezek/credit-card-anomaly-detection.git

cd credit-card-anomaly-detection

python3.10 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python src/anomaly_detection.py
```
