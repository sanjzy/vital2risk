import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

st.set_page_config(page_title="Vitals2Risk: Dual Model", layout="centered")
st.title('Vitals2Risk: Predict ICU Instability (RF vs LSTM)')

# Model selector
model_choice = st.selectbox("Choose model to predict risk:", ["Random Forest", "LSTM (Deep Learning)"])

uploaded = st.file_uploader("Upload CSV with columns HR, SPO2, RESP", type="csv")

# LSTM model definition
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=15, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1]))

if uploaded:
    df = pd.read_csv(uploaded)
    df = df.interpolate().dropna()

    if model_choice == "Random Forest":
        model = joblib.load("rf_model.joblib")
        for col in ['HR', 'SPO2', 'RESP']:
            df[col+'_mean'] = df[col].rolling(180).mean()
            df[col+'_std'] = df[col].rolling(180).std()
            df[col+'_min'] = df[col].rolling(180).min()
            df[col+'_max'] = df[col].rolling(180).max()

        df = df.dropna()
        features = df[[c for c in df.columns if any(stat in c for stat in ['mean','std','min','max'])]]
        preds = model.predict_proba(features)[:, 1]
        df = df.loc[features.index]
        df['Risk'] = preds

    else:  # LSTM
        model = LSTMModel()
        model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device('cpu')))
        model.eval()

        X_seq = []
        for i in range(180, len(df)):
            window = df.iloc[i-180:i]
            features = []
            for col in ['HR', 'SPO2', 'RESP']:
                features.append(window[col].values)
                features.append(window[col].rolling(5).mean().fillna(method='bfill').values)
                features.append(window[col].rolling(5).std().fillna(method='bfill').values)
                features.append(window[col].rolling(5).min().fillna(method='bfill').values)
                features.append(window[col].rolling(5).max().fillna(method='bfill').values)
            sample = np.stack(features, axis=1)
            X_seq.append(sample)

        X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
        with torch.no_grad():
            preds = model(X_tensor).squeeze().numpy()

        df = df.iloc[180:]
        df['Risk'] = preds

    # 📈 Show risk chart
    st.subheader("📊 Predicted Instability Risk")
    st.line_chart(df['Risk'])

    # ⚠️ High risk alert
    if (df['Risk'] > 0.85).any():
        st.error("⚠️ Warning: High instability risk detected!")

    # 📋 Final vitals + risk
    st.subheader("📋 Last 10 Predictions")
    st.dataframe(df[['HR', 'SPO2', 'RESP', 'Risk']].tail(10))

    # 📉 Timeline plot with HR
    st.subheader("📉 Timeline: HR vs Risk")
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(df.index, df['Risk'], 'r-', label='Risk')
    ax2.plot(df.index, df['HR'], 'b--', alpha=0.4, label='HR')
    ax1.set_ylabel("Risk")
    ax2.set_ylabel("HR")
    st.pyplot(fig)

    # ⬇️ CSV download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "risk_predictions.csv", "text/csv")
