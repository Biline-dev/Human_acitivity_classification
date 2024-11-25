import streamlit as st
from interface.processing import load_data_multi_modal, load_data_unified_modal
# Import your data processing function

import pandas as pd
# Streamlit app
st.title("Test Unified or Multimodal Model with Uploaded Datasets")

# Step 1: Select the Model
model_type = st.selectbox("Select the Model Type", options=["Unified", "Multimodal"])

# Step 2: Upload Multiple CSV Files
st.write("Upload the required dataset files:")
uploaded_files = st.file_uploader(
    "Upload exactly 4 CSV files in the correct order (EMG, IMU, IPS, Mocap)",
    type=["csv"],
    accept_multiple_files=True
)

# Ensure exactly 4 files are uploaded
if uploaded_files and len(uploaded_files) == 4:
    st.success("All 4 files uploaded successfully!")
    st.write("Preview of uploaded files:")

    # Preview each file
    files_path = []
    for i, file in enumerate(uploaded_files):
        st.write(f"File {i+1}: {file.name}")
        df = pd.read_csv(file, index_col=0)
        st.write(df.head())  # Display the first few rows
        files_path.append(file.name)

    # Step 3: Run Evaluation
    if st.button("Run Evaluation"):
        try:
            st.write(f"Running evaluation using the **{model_type}** model...")
            if model_type == "Unified":
                # Call your data processing function with the list of file paths
                y_pred = load_data_unified_modal(files_path)
            else:
                # Call your data processing function with the list of file paths
                y_pred = load_data_multi_modal(files_path)

            # Display the predictions
            st.write("Predictions:")
            st.write(y_pred)

            st.success("Evaluation completed successfully!")
        except Exception as e:
            st.error(f"Error during evaluation: {e}")

elif uploaded_files and len(uploaded_files) != 4:
    st.error("Please upload exactly 4 CSV files.")
