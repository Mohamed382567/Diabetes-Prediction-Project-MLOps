import streamlit as st
import pandas as pd
import numpy as np
import joblib

# CRITICAL FIX: Define EPSILON here to prevent NameError
EPSILON = 1e-6

# =========================================================
# 1. Project Setup & Disclaimer
# =========================================================
st.set_page_config(
    page_title="Diabetes Prediction System (Educational)",
    page_icon="🩺",
    layout="wide"
)

# --- Sidebar: Project Context and Disclaimer ---
with st.sidebar:
    st.header("ℹ️ Project Context & Disclaimer")
    st.info("""
    **⚠️ Educational MLOps Project**
    
    This application demonstrates an end-to-end Machine Learning pipeline (Data Cleaning -> Feature Engineering -> Modeling -> Deployment).
    
    **Important Note:**
    * Predictions are based on a limited dataset (~768 records) and are for **demonstration purposes only**.
    * This is **NOT a medical device** and should not replace professional medical advice.
    
    **🚀 Future Scalability:**
    The system requires significantly larger datasets and integration with professional healthcare systems for real-world use.
    """)
    st.write("---")
    st.write("Developed for MLOps Portfolio Demonstration.") 

# =========================================================
# 2. Title and Introduction
# =========================================================
st.title("🩺 Diabetes Risk Prediction System")
st.markdown("""
An intelligent system to assess the probability of diabetes using advanced Random Forest modeling and Feature Engineering.
""")
st.write("---")

# =========================================================
# 3. Model Loading
# =========================================================
@st.cache_resource
def load_models():
    try:
        # Load the saved models
        model = joblib.load('random_forest_model.joblib')
        imputer = joblib.load('iterative_imputer.joblib')
        scaler = joblib.load('standard_scaler.joblib')
        
        # Load training columns to ensure order
        try:
            model_cols = joblib.load('training_columns.joblib')
        except:
            # Fallback columns list (as derived from your notebook)
            model_cols = [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age',
                'Is_Glucose_Missing', 'Is_BloodPressure_Missing', 'Is_SkinThickness_Missing', 
                'Is_Insulin_Missing', 'Is_BMI_Missing', 'Log_DPF', 'Glucose_to_Insulin_Ratio', 
                'Age_BMI_Interaction', 'Sqrt_Insulin', 'Sqrt_Pregnancies', 'BP_Age_Index', 
                'Skin_BMI_Ratio', 'Is_Glucose_Critical', 
                'BMI_Category_Normal', 'BMI_Category_Obese_Class_I', 
                'BMI_Category_Obese_Class_II', 'BMI_Category_Obese_Class_III', 
                'BMI_Category_Overweight', 'BMI_Category_Underweight'
            ]
        return model, imputer, scaler, model_cols
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, imputer, scaler, model_columns = load_models()

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
# 5. Processing & Prediction Logic
# =========================================================
if st.button("🔍 Analyze Risk"):
    if model is not None:
        try:
            # --- A. Initial DataFrame Setup ---
            df = pd.DataFrame({
                'Pregnancies': [pregnancies], 'Glucose': [glucose], 'BloodPressure': [bp],
                'SkinThickness': [skin], 'Insulin': [insulin], 'BMI': [bmi],
                'DiabetesPedigreeFunction': [dpf], 'Age': [age]
            })
            
            # --- B. MICE Preparation and Imputation FIX ---
            target_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            
            # 1. Create Missing Flags and Convert 0s to NaN
            for col in target_cols:
                df[f'Is_{col}_Missing'] = (df[col] == 0).astype(int)
                df[col] = df[col].replace(0, np.nan)

            # 2. Define MICE Input Columns (FIXED: must include missing flags)
            mice_input_cols = [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                'DiabetesPedigreeFunction', 'Age'
            ] + [f'Is_{col}_Missing' for col in target_cols]
            
            # 3. Apply Imputer (passing exactly 13 columns)
            df[mice_input_cols] = imputer.transform(df[mice_input_cols])

            # Clamping Insulin (must be done after imputation if it filled the zero/NaN)
            df['Insulin'] = df['Insulin'].apply(lambda x: max(x, 1.0))
            
            # --- C. Feature Engineering (Your Custom Features) ---
            
            df['Log_DPF'] = np.log(df['DiabetesPedigreeFunction'].replace(0, EPSILON))
            df['Glucose_to_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + EPSILON)
            df['Age_BMI_Interaction'] = df['Age'] * df['BMI']
            df['Sqrt_Insulin'] = np.sqrt(df['Insulin'])
            df['Sqrt_Pregnancies'] = np.sqrt(df['Pregnancies'])
            df['BP_Age_Index'] = df['BloodPressure'] / (df['Age'] + EPSILON)
            df['Skin_BMI_Ratio'] = df['SkinThickness'] / (df['BMI'] + EPSILON)
            df['Is_Glucose_Critical'] = (df['Glucose'] >= 126).astype(int)

            # BMI Categories (Manual One-Hot Encoding)
            bmi_val = df['BMI'].iloc[0]
            bmi_cat = ""
            if bmi_val < 18.5: bmi_cat = "Underweight"
            elif 18.5 <= bmi_val < 25: bmi_cat = "Normal"
            elif 25 <= bmi_val < 30: bmi_cat = "Overweight"
            elif 30 <= bmi_val < 35: bmi_cat = "Obese_Class_I"
            elif 35 <= bmi_val < 40: bmi_cat = "Obese_Class_II"
            else: bmi_cat = "Obese_Class_III"

            expected_cats = ['Normal', 'Obese_Class_I', 'Obese_Class_II', 'Obese_Class_III', 'Overweight', 'Underweight']
            for cat in expected_cats:
                col_name = f"BMI_Category_{cat}"
                df[col_name] = 1 if cat == bmi_cat else 0

            # **CRITICAL FIX:** Drop original DPF column after creating Log_DPF
            if 'DiabetesPedigreeFunction' in df.columns:
                df = df.drop(columns=['DiabetesPedigreeFunction'])
            
            # --- D. Final Column Alignment and Scaling ---
            df_final = pd.DataFrame()
            for col in model_columns:
                if col in df.columns:
                    df_final[col] = df[col]
                else:
                    # Failsafe for missing columns (should be 0 for one-hot/flags)
                    df_final[col] = 0 
            
            # Scaling (Ensure only numerical columns are scaled)
            numerical_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 
                              'Log_DPF', 'Glucose_to_Insulin_Ratio', 'Age_BMI_Interaction', 'Sqrt_Insulin', 
                              'Sqrt_Pregnancies', 'BP_Age_Index', 'Skin_BMI_Ratio']
            
            scaling_cols = [col for col in numerical_cols if col in df_final.columns]

            if scaling_cols:
                df_final[scaling_cols] = scaler.transform(df_final[scaling_cols])
            
            # --- E. Prediction ---
            prediction = model.predict(df_final)[0]
            probability = model.predict_proba(df_final)[0][1]

            # --- F. Display Results ---
            st.write("---")
            st.subheader("📊 Prediction Result")
            
            if prediction == 1:
                st.error(f"**High Risk of Diabetes Detected**")
                st.write(f"Probability: **{probability*100:.1f}%**")
                st.progress(int(probability*100))
                st.warning("⚠️ This result suggests a high probability. Please consult a healthcare professional for accurate diagnosis.")
            else:
                st.success(f"**Low Risk Detected**")
                st.write(f"Probability: **{probability*100:.1f}%**")
                st.progress(int(probability*100))
                st.balloons()

            with st.expander("See Calculated Features (Engineered Data)"):
                st.dataframe(df_final.T)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Debugging Tip: Ensure that the columns in training_columns.joblib match the final features created here.")
    else:
        st.warning("Models not loaded. Check the paths and file names.")
