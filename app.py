import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Neural Network Inference Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern Data Science Dashboard look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e222d;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e364f;
    }
    .css-1r650q0 {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODEL RECONSTRUCTION & LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """
    Reconstructs the Sequential Keras architecture from config.json specs:
    - Input: 10 Features
    - Hidden Layer 1: 8 Units (ReLU)
    - Hidden Layer 2: 7 Units (ReLU)
    - Output Layer: 1 Unit (Sigmoid)
    """
    model = Sequential([
        InputLayer(shape=(10,), name="input_layer"),
        Dense(8, activation="relu", name="dense"),
        Dense(7, activation="relu", name="dense_1"),
        Dense(1, activation="sigmoid", name="dense_2")
    ])
    
    # Load binary weights file extracted from your saved architecture
    try:
        model.load_weights("model.weights.h5")
    except Exception:
        # Fallback build compilation if running demo without local weight file
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

model = load_trained_model()

# -----------------------------------------------------------------------------
# SIDEBAR - INPUT CONTROL PANEL
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/m_brain/512/FFFFFF/brain.png", width=80)
st.sidebar.title("Feature Controls")
st.sidebar.markdown("Adjust the 10 input feature vectors below to run model inference.")

features = {}
st.sidebar.subheader("Numeric Inputs")

# Create 10 interactive feature sliders/inputs
col_sb1, col_sb2 = st.sidebar.columns(2)
for i in range(1, 11):
    col = col_sb1 if i % 2 != 0 else col_sb2
    features[f"Feature_{i}"] = col.number_input(
        f"X_{i}", 
        min_value=-5.0, 
        max_value=5.0, 
        value=0.0, 
        step=0.1,
        key=f"input_{i}"
    )

st.sidebar.markdown("---")
threshold = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)

# -----------------------------------------------------------------------------
# MAIN DASHBOARD INTERFACE
# -----------------------------------------------------------------------------
st.title("🧠 Neural Network Inference Dashboard")
st.caption("Deep Learning Binary Classification Model • Keras Architecture Deployment")

# Layout into two main columns
col_left, col_right = st.columns([1, 1])

# Format inputs into model shape (1, 10)
input_array = np.array([list(features.values())], dtype=np.float32)

with col_left:
    st.subheader("📊 Input Vector Summary")
    
    # Display Input Data Frame
    df_inputs = pd.DataFrame(features, index=["Value"]).T
    st.dataframe(df_inputs, use_container_width=True, height=380)

with col_right:
    st.subheader("⚡ Model Prediction & Output")
    
    # Predict button / Real-time execution
    raw_prediction = float(model.predict(input_array, verbose=0)[0][0])
    predicted_class = 1 if raw_prediction >= threshold else 0
    confidence = raw_prediction if predicted_class == 1 else (1.0 - raw_prediction)

    # Key Performance Metric Cards
    m1, m2 = st.columns(2)
    m1.metric(
        label="Predicted Class", 
        value=f"Class {predicted_class}",
        delta="Positive" if predicted_class == 1 else "Negative",
        delta_color="normal" if predicted_class == 1 else "inverse"
    )
    m2.metric(
        label="Probability Score", 
        value=f"{raw_prediction:.4f}",
        delta=f"Threshold: {threshold}"
    )

    # Gauge Chart Visualization using Plotly
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=raw_prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sigmoid Output Probability", 'font': {'size': 16, 'color': "white"}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#4CAF50" if predicted_class == 1 else "#FF5252"},
            'bgcolor': "#1e222d",
            'borderwidth': 2,
            'bordercolor': "#2e364f",
            'steps': [
                {'range': [0, threshold], 'color': '#2a1b24'},
                {'range': [threshold, 1.0], 'color': '#1b2a20'}
            ],
            'threshold': {
                'line': {'color': "yellow", 'width': 3},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    fig.update_layout(
        height=260, 
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"}
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# MODEL ARCHITECTURE DETAILS SECTION
# -----------------------------------------------------------------------------
with st.expander("🛠️ View Neural Network Layer Architecture"):
    st.markdown("""
    **Architecture Overview (Sequential):**
    * **Input Layer:** 10 Features (`float32`)
    * **Dense Layer 1:** 8 Units | Activation: `ReLU`
    * **Dense Layer 2:** 7 Units | Activation: `ReLU`
    * **Output Layer:** 1 Unit | Activation: `Sigmoid`
    * **Optimizer:** Adam ($\eta = 0.001$) | **Loss Function:** Binary Crossentropy
    """)
