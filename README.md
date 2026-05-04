# 🌾 Smart Crop Recommendation System

This project is an AI-based Smart Agriculture application that recommends the most suitable crop based on soil nutrients and environmental conditions.

## 🚀 Features
- Predicts best crop using Machine Learning
- Uses soil data (N, P, K) and environmental data (temperature, humidity, pH, rainfall)
- Built with Random Forest algorithm
- Interactive user interface using Streamlit
- Displays feature correlation heatmap
- Shows prediction with model accuracy

## 🧠 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

## 📊 How It Works
1. User enters soil and climate values
2. Data is preprocessed and scaled
3. Machine learning model predicts best crop
4. Result is displayed along with accuracy

## ▶️ Run the Project

```bash
pip install streamlit
streamlit run crop_recommendation.py
