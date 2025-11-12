import streamlit as st
st.set_page_config(
    page_title="Flood Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import ssl

# Ignore SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# -------------------- DATA LOADING --------------------
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

df = load_data(r'D:\sejong_major\projects\flood-prediction\Project\Data.csv')

# -------------------- SIDEBAR MENU --------------------
st.sidebar.header("⚙️ Dashboard Navigation")
menu = st.sidebar.radio("Menu", ["🏠 Home", "📊 Data Overview", "🧠 Model Training", "📈 Correlation"])

# -------------------- HOME PAGE --------------------
if menu == "🏠 Home":
    st.title("🌊 Flood Prediction System")
    st.markdown("A machine learning dashboard for analyzing and predicting flood data.")
    st.image("https://cdn-icons-png.flaticon.com/512/4149/4149670.png", width=200)
    st.write("---")
    st.header("🔹 Dataset Preview")
    st.dataframe(df.head())

# -------------------- RAW DATA PAGE --------------------
elif menu == "📊 Data Overview":
    st.title("📊 Dataset Exploration")
    st.dataframe(df)

    st.subheader("Train-Test Split Ratio")
    ratio = st.select_slider("Choose Train-Test Split", options=[0.6, 0.7, 0.8, 0.9], value=0.8)
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - ratio, random_state=42)
    
    col1, col2 = st.columns(2)
    col1.metric("Training Samples", X_train.shape[0])
    col2.metric("Testing Samples", X_test.shape[0])
    
    with st.expander("📋 View Training Data"):
        st.dataframe(X_train)
    with st.expander("📋 View Testing Data"):
        st.dataframe(X_test)

# -------------------- MODEL TRAINING --------------------
elif menu == "🧠 Model Training":
    st.title("🧠 Model Training and Evaluation")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    test_ratio = st.slider("Select Test Size", 0.1, 0.5, 0.2, step=0.05)
    n_estimators = st.slider("Number of Trees (Estimators)", 50, 500, 100, step=50)
    random_state = st.number_input("Random State", value=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
    
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.write("### ✅ Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.3f}")
    col3.metric("RMSE", f"{rmse:.3f}")
    

    # Feature Importance Plot
    st.write("### 🔍 Feature Importance")
    importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=importances, x="Importance", y="Feature", palette="viridis", ax=ax)
    st.pyplot(fig)

# -------------------- CORRELATION --------------------
elif menu == "📈 Correlation":
    st.title("📈 Correlation Analysis")
    st.markdown("Visualize the relationships between features using a heatmap.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    st.pyplot(fig)

# -------------------- STYLE TWEAKS --------------------
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stMetric {background-color: #f8f9fa; border-radius: 10px; padding: 15px;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
