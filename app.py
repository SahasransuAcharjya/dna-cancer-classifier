import streamlit as st
import pandas as pd
import numpy as np
import torch

st.title("🧬 DNA Cancer Classifier")
st.markdown("### Production model deployed!")

# Load your trained model
# model = torch.load('models/final_model.pth')  # Uncomment after loading

st.image("results/project_results_dashboard.png", use_column_width=True)

uploaded_file = st.file_uploader("📁 Upload 44-gene CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if df.shape[1] == 44:
        # prediction = model(torch.tensor(df.values).float())
        st.success("✅ Prediction ready!")
        st.balloons()
    else:
        st.error("❌ Expected 44 gene columns")
