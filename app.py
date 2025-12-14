import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# CRITICAL FIX: Define EPSILON globally for use in feature engineering
EPSILON = 1e-6
# Define the custom threshold used during model training
CUSTOM_THRESHOLD = 0.40

# --- Helper Functions (Copied from src/features/build_features.py for Streamlit) ---
GLUCOSE_CRITICAL_CUTOFF = 126

def classify_bmi(bmi: float) -> str:
    """Classifies BMI into standard WHO categories."""
    if bmi < 18.5: return 'Underweight'
    elif 18.5 <= bmi < 25: return 'Normal'
    elif 25 <= bmi < 30: return 'Overweight'
    elif 30 <= bmi < 35: return 'Obese_Class_I'
    elif 35 <= bmi < 40: return 'Obese_Class_II'
    else: return 'Obese_Class_III'

# =========================================================
# 1. Project Setup & Configuration
# =========================================================
st.set_page_config(
    page_title="Diabetes Prediction System (Educational)",
    page_icon="🩺",
    layout="wide"
)

# --- Sidebar: Project Context and Disclaimer ---
with st.sidebar:
    st.header("ℹ️ Project Context & Disclaimer")
    st.info(f"""
    **⚠️ Educational MLOps Project**
    This application demonstrates an end-to-end Machine Learning pipeline.
    
    **Important Note:**
    * Predictions are based on a limited dataset and are for **demonstration purposes only**.
    * This is **NOT a medical device** and should not replace professional medical advice.
    * **Decision Threshold:** The model uses a {CUSTOM_THRESHOLD*100:.0f}% threshold.
    
    **🚀 Future Scalability:**
    The system requires significantly larger datasets and integration with professional healthcare systems for real-world use.
    """)
    st.write("---")
    st.write("Developed by: [Your Name]") # Update with your name

# =========================================================
# 2. Model Loading (Artifacts)
# =========================================================
@st.cache_resource
def load_models():
    """Loads the model and all preprocessors (Imputer, Scaler) and feature names."""
    try:
        # Assuming all artifacts are in the same folder as app.py for Streamlit Cloud deployment
        model = joblib.load('random_forest_model.joblib')
        imputer = joblib.load('training_columns.joblib')
        scaler = joblib.load('sandard_scaler.joblib')
        
        # CRITICAL: Load training columns to ensure order for prediction
        model_cols = joblib.load('training_columns.joblib')
        
        st.success("✅ All model artifacts loaded successfully.")
        return model, imputer, scaler, model_cols
    except Exception as e:
        st.error(f"Error loading models. Please ensure all artifacts (.pkl, .joblib) are uploaded correctly: {e}")
        st.stop()
        return None, None, None, None

model, imputer, scaler, model_columns = load_models()

# Show current file structure for debugging
st.write("---")
st.write("Current Directory:", os.getcwd())
st.write("Files in Directory:", os.listdir())
st.write("---")

# =========================================================
# 3. Data Processing Pipeline (Inference Mode)
# =========================================================
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all the custom feature generation logic exactly as in training."""
    df_eng = df.copy()

    # --- 1. Log and Sqrt Transforms ---
    df_eng['Log_DPF'] = np.log(df_eng['DiabetesPedigreeFunction'] + EPSILON)
    # Note: Log_Age not included in your app.py inputs but added here for completeness if you use it.
    df_eng['Log_Age'] = np.log1p(df_eng['Age']) 
    df_eng['Sqrt_Insulin'] = np.sqrt(df_eng['Insulin'].clip(lower=0))
    df_eng['Sqrt_Pregnancies'] = np.sqrt(df_eng['Pregnancies'].clip(lower=0))

    # --- 2. Ratios & Interactions ---
    df_eng['Glucose_to_Insulin_Ratio'] = df_eng['Glucose'] / (df_eng['Insulin'] + EPSILON)
    df_eng['Age_BMI_Interaction'] = df_eng['Age'] * df_eng['BMI']
    df_eng['BP_Age_Index'] = df_eng['BloodPressure'] / (df_eng['Age'] + EPSILON)
    df_eng['Skin_BMI_Ratio'] = df_eng['SkinThickness'] / (df_eng['BMI'] + EPSILON)

    # --- 3. Critical Flags & Categorization ---
    df_eng['Is_Glucose_Critical'] = (df_eng['Glucose'] >= GLUCOSE_CRITICAL_CUTOFF).astype(int)
    
    # Apply BMI Classification and One-Hot Encoding
    df_eng['BMI_Category'] = df_eng['BMI'].apply(classify_bmi)
    df_eng = pd.get_dummies(df_eng, columns=['BMI_Category'], drop_first=True) 

    # --- 4. Drop Original Columns ---
    if 'DiabetesPedigreeFunction' in df_eng.columns:
        df_eng = df_eng.drop('DiabetesPedigreeFunction', axis=1)
    
    df_eng.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df_eng

# =========================================================
# 4. User Inputs Interface
# =========================================================
st.subheader("📝 Patient Vitals Input")

col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120, help="Enter 0 if the value is unknown or missing.")
    bp = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70, help="Enter 0 if unknown.")

with col2:
    skin = st.number_input("Skin Thickness (mm)", 0, 100, 20, help="Enter 0 if unknown.")
    insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 79, help="Enter 0 if unknown.")
    bmi = st.number_input("BMI", 0.0, 70.0, 32.0, help="Enter 0 if unknown.")

with col3:
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47)
    age = st.number_input("Age (Years)", 1, 120, 33)

# =========================================================
# 5. Prediction Execution
# =========================================================
if st.button("🔍 Analyze Risk"):
    try:
        # --- A. Initial DataFrame Setup ---
        df = pd.DataFrame({
            'Pregnancies': [pregnancies], 'Glucose': [glucose], 'BloodPressure': [bp],
            'SkinThickness': [skin], 'Insulin': [insulin], 'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf], 'Age': [age]
        })
        
        # --- B. MICE Preparation and Imputation ---
        target_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        # 1. Create Missing Flags and Convert 0s to NaN (Pre-Imputer steps)
        for col in target_cols:
            df[f'Is_{col}_Missing'] = (df[col] == 0).astype(int)
            df[col] = df[col].replace(0, np.nan)

        # 2. Define MICE Input Columns (must match Imputer fit)
        mice_input_cols = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
            'DiabetesPedigreeFunction', 'Age'
        ] + [f'Is_{col}_Missing' for col in target_cols]
        
        # 3. Apply Imputer (Transforms only the MICE input columns)
        df[mice_input_cols] = imputer.transform(df[mice_input_cols])

        # Clamping Insulin (if necessary after imputation)
        df['Insulin'] = df['Insulin'].apply(lambda x: max(x, 1.0))
        
        # --- C. Feature Engineering ---
        df_engineered = apply_feature_engineering(df)
        
        # --- D. Final Column Alignment (CRITICAL FIX for One-Hot Encoding) ---
        # This aligns the current features to the exact feature set the model expects,
        # adding missing columns (like Obese_Class_I if not present) with 0.
        df_final = df_engineered.reindex(columns=model_columns, fill_value=0)
        
        # --- E. Scaling ---
        # The list of numerical columns must match the columns scaled during training
        scaling_cols_for_transform = [
             'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 
             'Log_DPF', 'Glucose_to_Insulin_Ratio', 'Age_BMI_Interaction', 'Sqrt_Insulin', 
             'Sqrt_Pregnancies', 'BP_Age_Index', 'Skin_BMI_Ratio'
        ]

        # Filter to ensure only columns present in df_final are scaled
        scaling_cols_for_transform = [col for col in scaling_cols_for_transform if col in df_final.columns]

        if scaling_cols_for_transform:
            df_final[scaling_cols_for_transform] = scaler.transform(df_final[scaling_cols_for_transform])
        
        # --- F. Prediction Logic ---
        probability = model.predict_proba(df_final)[0][1]
        
        # Apply the 40% custom threshold
        if probability >= CUSTOM_THRESHOLD:
            prediction = 1
        else:
            prediction = 0

        # --- G. Display Results ---
        st.write("---")
        st.subheader("📊 Prediction Result")
        
        if prediction == 1:
            st.error(f"**High Risk of Diabetes Detected**")
            st.write(f"Probability: **{probability*100:.1f}%** (Decision Threshold: {CUSTOM_THRESHOLD*100:.0f}%)")
            st.progress(int(probability*100))
            st.warning("⚠️ This result suggests a high probability. Please consult a healthcare professional for accurate diagnosis.")
        else:
            st.success(f"**Low Risk Detected**")
            st.write(f"Probability: **{probability*100:.1f}%** (Decision Threshold: {CUSTOM_THRESHOLD*100:.0f}%)")
            st.progress(int(probability*100))
            st.balloons()

        with st.expander("See Calculated Features (Engineered Data)"):
            st.dataframe(df_final.T)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Debugging Hint: Check if the artifacts (model.pkl, columns.pkl, etc.) were generated from the correct training script.")

else:
    st.warning("Models not loaded. Check the paths and file names.")

