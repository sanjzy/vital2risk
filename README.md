# Vitals2Risk: ICU Instability Predictor

Vitals2Risk is a machine learning + deep learning web app that predicts patient instability using ICU vital signs.

🔗 **Live App:** [Click here to try it!](https://<your-link>.streamlit.app)

## 🚀 Features

- 📤 Upload CSVs with HR, SPO2, RESP
- 🧠 Choose between Random Forest or LSTM
- 📈 See real-time risk prediction charts
- ⚠️ Get high-risk alerts (threshold > 0.85)
- 📉 Timeline chart: HR vs Instability Risk
- 📥 Download results as CSV

## 🧠 Models Used
- **Random Forest:** trained on engineered rolling stats
- **LSTM:** trained on time series windows with 15 features

## 🛠 Tech Stack
- Python, Streamlit
- scikit-learn, PyTorch
- Matplotlib, Pandas, Numpy

## 🩺 Use Case
Used in ICU monitoring to flag early warning signs for:
- Tachycardia (HR > 110)
- Hypoxia (SPO2 < 92)
- Respiratory distress (RESP > 28)

## 📁 Example CSV format

| HR | SPO2 | RESP |
|----|------|------|
| 85 | 97   | 20   |
| 88 | 96   | 22   |
| 91 | 95   | 25   |

## 👨‍💻 Developed by
**Sanjay C** — BSc Bioinformatics Final Year  

