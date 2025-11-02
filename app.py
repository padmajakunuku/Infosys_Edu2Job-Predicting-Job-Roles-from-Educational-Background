import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Page Configuration (Wide Layout, New Icon) ---
# This MUST be the first Streamlit command
st.set_page_config(
    page_title="Job Role Predictor",
    page_icon="✨",
    layout="wide"
)

# --- 2. Load the Saved Model, Columns, and Encoder ---
# (Same as before)
try:
    model = joblib.load('random_forest_model.joblib')
    model_columns = joblib.load('model_columns.joblib')
    encoder = joblib.load('label_encoder.joblib')
except FileNotFoundError:
    st.error("Error: Model/column/encoder files not found. Make sure all 3 .joblib files are in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading files: {e}")
    st.stop()

# --- 3. Define the Input Options (Same as before) ---
SKILL_1_OPTIONS = [
    'AI', 'Accounting', 'Adobe Photoshop', 'Anatomy', 'Attention to Detail', 'Behavioral Analysis', 'Biology', 'Brand Management', 'Budgeting', 'C++', 'CAD Design', 'Case Analysis', 'Chemistry', 'Circuit Design', 'Classroom Management', 'Clinical Practice', 'Cloud Computing', 'Communication', 'Conflict Resolution', 'Construction Planning', 'Counseling', 'Creative Thinking', 'Creativity', 'Curriculum Design', 'Cybersecurity', 'Data Analysis', 'Data Structures', 'Deep Learning', 'Documentation', 'Educational Research', 'Electronics', 'Embedded Systems', 'Emergency Handling', 'Ethical Hacking', 'Excel', 'Financial Analysis', 'Forecasting', 'Graphic Design', 'HR Management', 'Illustration', 'Java', 'Laboratory Skills', 'Leadership', 'Legal Research', 'Linux', 'Machine Learning', 'Marketing Strategy', 'Materials Science', 'Medical Diagnosis', 'Negotiation', 'Network Security', 'Networking', 'Nursing', 'Patient Care', 'Pharmacology', 'Problem Solving', 'Project Management', 'Project Planning', 'Public Speaking', 'Publication Writing', 'Python', 'Recruitment', 'Research', 'Risk Assessment', 'Routing', 'SEO', 'SQL', 'Scheduling', 'Social Media', 'Statistics', 'Structural Design', 'Subject Expertise', 'Surgery', 'Surveying', 'Switching', 'Teaching', 'Thermodynamics', 'Visualization'
]
SKILL_2_OPTIONS = SKILL_1_OPTIONS.copy()
EDUCATION_OPTIONS = [
    'Bachelor of Architecture', 'Bachelor of Arts', 'Bachelor of Commerce', 'Bachelor of Engineering', 'Bachelor of Law', 'Bachelor of Medicine and Surgery', 'Bachelor of Nursing', 'Bachelor of Science', 'MBA', 'Master of Architecture', 'Master of Arts', 'Master of Business Administration', 'Master of Engineering', 'Master of Law', 'Master of Medicine', 'Master of Nursing', 'Master of Science', 'PhD in Biology', 'PhD in Computer Science', 'PhD in Education', 'PhD in Psychology'
]

# --- 4. Streamlit UI ---  <- THIS SECTION IS MODIFIED

# Title and introduction (Centered using columns)
_col1, mid_col, _col3 = st.columns([1, 2, 1]) # Create 3 columns, middle one is 2x wide
with mid_col:
    # Title is in the middle column
    st.title("✨ Job Role Predictor ✨") 
    
# Center the markdown text using HTML (the only way to truly center)
st.markdown("<p style='text-align: center;'>This app predicts your job role using a Random Forest model. Fill in the form below to get your prediction!</p>", unsafe_allow_html=True)


# --- 5. Input Form (This is the new, cleaner layout) ---
with st.form(key="prediction_form"):
    st.subheader("Enter Your Details:")
    
    # --- Create 3 columns ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_skill_1 = st.selectbox(
            "Select your Primary Skill:",
            options=SKILL_1_OPTIONS,
            index=None,
            placeholder="Choose a primary skill..."
        )
    
    with col2:
        selected_skill_2 = st.selectbox(
            "Select your Secondary Skill:",
            options=SKILL_2_OPTIONS,
            index=None,
            placeholder="Choose a secondary skill..."
        )
    
    with col3:
        selected_education = st.selectbox(
            "Select your Educational Qualification:",
            options=EDUCATION_OPTIONS,
            index=None,
            placeholder="Choose your qualification..."
        )
    
    st.markdown("---") # Visual separator
    
    # --- Form Submit Button ---
    submit_button = st.form_submit_button(
        label="Predict Job Role",
        type="primary",
        use_container_width=True
    )

# --- 6. Prediction Logic (Runs *after* the form is submitted) ---
if submit_button:
    # Check if all inputs are filled
    if not selected_skill_1 or not selected_skill_2 or not selected_education:
        st.warning("Please fill in all three fields to make a prediction.")
    else:
        try:
            # Create the input dataframe for the model (same logic)
            input_data = pd.DataFrame(columns=model_columns)
            input_data.loc[0] = 0
            
            col_skill_1 = f"Skill_1_{selected_skill_1}"
            col_skill_2 = f"Skill_2_{selected_skill_2}"
            col_education = f"Educational_Qualifications_{selected_education}"
            
            input_data[col_skill_1] = 1
            input_data[col_skill_2] = 1
            input_data[col_education] = 1
            
            # Make prediction
            prediction_numeric = model.predict(input_data)
            predicted_role = encoder.inverse_transform(prediction_numeric)
            
            # --- Display the result in a cleaner way ---
            st.subheader(f"🚀 Your Predicted Job Role is:")
            st.success(f"**{predicted_role[0]}**")
            st.balloons()
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- 7. Sidebar (Cleaned up) ---
st.sidebar.header("How to Use")
st.sidebar.info(
    "1. Select your top two skills.\n"
    "2. Select your highest education.\n"
    "3. Click the 'Predict Job Role' button.\n"
    "4. The model will predict your job role!"
)

# --- 8. Expander (To hide extra info) ---
with st.expander("About this App"):
    st.info(
        "This app uses a Random Forest model trained on your custom dataset. "
    )