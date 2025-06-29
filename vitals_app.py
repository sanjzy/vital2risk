import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Vitals2Risk", layout="centered")

st.title('Vitals2Risk: Predict ICU Instability')
uploaded = st.file_uploader("Upload CSV with columns HR, SPO2, RESP", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    model = joblib.load("rf_model.joblib")

    for col in ['HR', 'SPO2', 'RESP']:
        df[col + '_mean'] = df[col].rolling(180).mean()
        df[col + '_std'] = df[col].rolling(180).std()
        df[col + '_min'] = df[col].rolling(180).min()
        df[col + '_max'] = df[col].rolling(180).max()

    df = df.dropna()
    features = df[[c for c in df.columns if any(stat in c for stat in ['mean','std','min','max'])]]
    preds = model.predict_proba(features)[:, 1]
    df['Risk'] = preds

    # 📈 Show risk chart
    st.subheader("📊 Predicted Instability Risk")
    st.line_chart(df['Risk'])

    # 🔔 High-risk alert
    if (preds > 0.85).any():
        st.error("⚠️ Warning: High instability risk detected!")

    # 📋 Show last 10 rows of predictions
    st.subheader("📋 Final Vitals + Risk")
    st.dataframe(df[['HR', 'SPO2', 'RESP', 'Risk']].tail(10))

    # 🕒 Timeline chart with vitals
    st.subheader("📉 Timeline Chart: Risk vs HR")
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(df.index, df['Risk'], 'r-', label='Risk')
    ax2.plot(df.index, df['HR'], 'b--', alpha=0.4, label='HR')
    ax1.set_ylabel('Risk')
    ax2.set_ylabel('HR')
    st.pyplot(fig)

    # 📤 Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predicted Risk as CSV", csv, "predicted_risk.csv", "text/csv")
