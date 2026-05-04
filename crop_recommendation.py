import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(layout="wide")

st.title("🌾 Crop Recommendation")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("crop_recommendation.csv")

X = df.drop('label', axis=1)
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ===============================
# LAYOUT (IMPORTANT - FIRST)
# ===============================
col1, col2 = st.columns([1.5, 1])

# ===============================
# INPUT SECTION
# ===============================
with col1:
    st.subheader("Input Values")

    c1, c2, c3 = st.columns(3)

    with c1:
        N = st.number_input("N", min_value=0.0)
        temp = st.number_input("Temperature")
    with c2:
        P = st.number_input("P", min_value=0.0)
        humidity = st.number_input("Humidity")
    with c3:
        K = st.number_input("K", min_value=0.0)
        ph = st.number_input("pH")

    rainfall = st.number_input("Rainfall")

    # ===============================
    # PREDICT BUTTON + OUTPUT BELOW
    # ===============================
    if st.button("🌿 Predict Crop"):

        input_data = pd.DataFrame(
            [[N, P, K, temp, humidity, ph, rainfall]],
            columns=X.columns
        )

        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)
        crop = le.inverse_transform(pred)[0]

        acc = accuracy_score(y_test, model.predict(X_test))

        # OUTPUT BELOW BUTTON
        st.success(f"🌟 Recommended Crop: {crop}")
        st.info(f"🎯 Accuracy: {acc*100:.2f}%")

# ===============================
# HEATMAP SECTION
# ===============================
with col2:
    st.subheader("Heatmap")

    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(df.corr(numeric_only=True), cmap="Greens", ax=ax, cbar=False)
    ax.tick_params(labelsize=6)

    st.pyplot(fig)