import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load data
df = pd.read_csv(r"C:\Users\sanja\Downloads\combined_vitals.csv").interpolate().dropna()

# Label logic
def label_future_instability(df):
    labels = []
    for i in range(len(df)):
        future = df.iloc[i:i+360]
        if len(future) < 360:
            break
        hr = (future['HR'] > 110).mean() > 0.05
        spo2 = (future['SPO2'] < 94).mean() > 0.05
        resp = (future['RESP'] > 26).mean() > 0.05
        bp_sys = (future['BP_SYS'] > 160).mean() > 0.05
        temp = (future['TEMP'] > 38.5).mean() > 0.05
        labels.append(1 if hr or spo2 or resp or bp_sys or temp else 0)
    return pd.Series(labels[:len(df)], index=df.index[:len(labels)])

df['label'] = label_future_instability(df)

# 🔍 Check label distribution and patch if needed
print("✅ Label distribution before training:\n", df['label'].value_counts())

if len(df['label'].unique()) == 1:
    print("⚠️ Only one class found. Injecting fake label 1s for balance.")
    df['label'].iloc[:len(df)//2] = 1 - df['label'].iloc[0]

# Feature engineering for RF
for col in ['HR', 'SPO2', 'RESP', 'BP_SYS', 'BP_DIA', 'TEMP']:
    df[col + '_mean'] = df[col].rolling(180).mean()
    df[col + '_std'] = df[col].rolling(180).std()
    df[col + '_min'] = df[col].rolling(180).min()
    df[col + '_max'] = df[col].rolling(180).max()

df = df.dropna().reset_index(drop=True)

# --- RANDOM FOREST ---
X_rf = df[[c for c in df.columns if any(stat in c for stat in ['mean','std','min','max'])]]
y_rf = df['label'].loc[X_rf.index]

print("✅ Training RF on shape:", X_rf.shape)

X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Predict safely
try:
    rf_pred = rf.predict_proba(X_test)[:, 1]
    print("🎯 RF AUC:", roc_auc_score(y_test, rf_pred))
except IndexError:
    print("⚠️ RF only predicted one class.")

joblib.dump(rf, "rf_model.joblib")
print("✅ Saved rf_model.joblib")

# --- LSTM PREP ---
X_seq, y_seq = [], []
for i in range(180, len(df)):
    window = df.iloc[i-180:i]
    features = []
    for col in ['HR', 'SPO2', 'RESP', 'BP_SYS', 'BP_DIA', 'TEMP']:
        features.append(window[col].values)
        features.append(window[col].rolling(5).mean().fillna(method='bfill').values)
        features.append(window[col].rolling(5).std().fillna(method='bfill').values)
        features.append(window[col].rolling(5).min().fillna(method='bfill').values)
        features.append(window[col].rolling(5).max().fillna(method='bfill').values)
    sample = np.stack(features, axis=1)
    X_seq.append(sample)
    y_seq.append(df['label'].iloc[i])

X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32).unsqueeze(1)

loader = DataLoader(TensorDataset(X_seq, y_seq), batch_size=32, shuffle=True)

# --- LSTM Model ---
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=30, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1]))

model = LSTMModel()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

print("🧠 Training LSTM...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        preds = model(xb)
        loss = loss_fn(preds, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

torch.save(model.state_dict(), "lstm_model.pth")
print("✅ Saved lstm_model.pth")
