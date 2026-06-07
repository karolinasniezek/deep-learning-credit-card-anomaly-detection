import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv("../data/creditcard.csv")

X = df.drop(["Time", "Class"], axis=1).values
y = df["Class"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train = X_scaled[y == 0]
X_test = X_scaled

input_dim = X_train.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation="relu")(input_layer)
encoded = BatchNormalization()(encoded)
encoded = Dropout(0.2)(encoded)

encoded = Dense(encoding_dim, activation="relu")(encoded)
print("encoded")
print(encoded)

decoded= Dense(32, activation="relu")(encoded)
decoded= BatchNormalization()(decoded)
decoded= Dropout(0.2)(decoded)

decoded = Dense(input_dim, activation="linear")(decoded)
print("Decoded")
print(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

autoencoder.fit(X_train, X_train,
                epochs=20,
                batch_size=256,
                shuffle=True,
                validation_split=0.1)

# Measure how much the reconstructed transactions differ from the original input data.
X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
print("Reconstruction errors (higher values may indicate anomalies):")
print(mse)

threshold = np.float64(np.percentile(mse[y == 0], 95))
y_pred = (mse > threshold).astype(int)
print(y_pred)

roc_auc = roc_auc_score(y, mse)
cr = classification_report(y, y_pred)
print(f"Anomaly Detection ROC-AUC: {roc_auc:.4f}")
print(f"\nClassification Report:\n{cr}")

# =====================
# ROC CURVE
# =====================

fpr, tpr, _ = roc_curve(y, mse)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"ROC Curve (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Anomaly Detection ROC Curve")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "../figures/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()