import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Setup premium layout
st.set_page_config(page_title="EduPro Dashboard", page_icon=":material/school:", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for rich aesthetics
st.markdown("""
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        transform: translateY(-2px);
    }
    .metric-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E40AF;
    }
    .metric-title {
        font-size: 1rem;
        font-weight: 600;
        color: #4B5563;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    enrollment_model = joblib.load('models/enrollment_model.joblib')
    revenue_model = joblib.load('models/revenue_model.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    try:
        enroll_fi = joblib.load('models/enrollment_fi.joblib')
        rev_fi = joblib.load('models/revenue_fi.joblib')
    except:
        enroll_fi, rev_fi = None, None
    return enrollment_model, revenue_model, feature_names, enroll_fi, rev_fi

@st.cache_data
def load_data():
    return pd.read_csv('processed_data.csv')

st.title(":material/monitoring: EduPro Demand & Revenue Forecast")

# Load assets
try:
    enrollment_model, revenue_model, feature_names, enroll_fi, rev_fi = load_models()
    df = load_data()
except Exception as e:
    st.error(f"Models or data not found. Please run the pipeline first. Error: {e}")
    st.stop()

# Dynamic options based on data
categories = df['category'].unique().tolist() if 'category' in df.columns else ['Tech', 'Business', 'Arts', 'Science']
levels = df['level'].unique().tolist() if 'level' in df.columns else ['Beginner', 'Intermediate', 'Advanced']

# Sidebar for inputs
st.sidebar.header("Configure Course Profile")
price = st.sidebar.slider("Course Price ($)", 10, 300, 50)
duration = st.sidebar.slider("Duration (Hours)", 1, 100, 10)
level = st.sidebar.selectbox("Course Level", levels)
category = st.sidebar.selectbox("Category", categories)

st.sidebar.markdown("---")
st.sidebar.header("Instructor Profile")
exp_years = st.sidebar.slider("Experience (Years)", 0, 40, 5)
teacher_rating = st.sidebar.slider("Teacher Rating", 1.0, 5.0, 4.5, 0.1)
course_rating = st.sidebar.slider("Expected Course Rating", 1.0, 5.0, 4.2, 0.1)

# Derived features (based on pipeline logic)
def get_price_band(p):
    if p <= 20: return 'Low'
    elif p <= 100: return 'Medium'
    else: return 'High'

def get_duration_bucket(d):
    if d <= 10: return 'Short'
    elif d <= 25: return 'Medium'
    elif d <= 50: return 'Long'
    else: return 'Very Long'

def get_rating_tier(r):
    if r <= 3.5: return 'Low'
    elif r <= 4.5: return 'Medium'
    else: return 'High'

def get_exp_bucket(e):
    if e <= 5: return 'Junior'
    elif e <= 10: return 'Mid'
    elif e <= 20: return 'Senior'
    else: return 'Expert'

input_df = pd.DataFrame([{
    'price': price,
    'duration_hours': duration,
    'level': level,
    'category': category,
    'experience_years': exp_years,
    'teacher_rating': teacher_rating,
    'course_rating': course_rating,
    'price_band': get_price_band(price),
    'duration_bucket': get_duration_bucket(duration),
    'rating_tier': get_rating_tier(course_rating),
    'experience_bucket': get_exp_bucket(exp_years),
    'match_score': 1 # Assuming expertise match for new course
}])

st.markdown("### Real-time Forecasting")

if st.button("Generate Forecast", use_container_width=True):
    with st.spinner("Calculating predictions..."):
        pred_enrollment = enrollment_model.predict(input_df)[0]
        pred_revenue = revenue_model.predict(input_df)[0]
        
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title"><span class="material-symbols-outlined" style="vertical-align: text-bottom;">group</span> Expected Enrollments</div>
            <div class="metric-value">{int(max(0, pred_enrollment)):,}</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title"><span class="material-symbols-outlined" style="vertical-align: text-bottom;">payments</span> Expected Revenue</div>
            <div class="metric-value">${max(0, pred_revenue):,.2f}</div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3 = st.tabs([":material/bar_chart: Feature Importance", ":material/pie_chart: Category Insights", ":material/account_balance_wallet: Revenue Analytics"])

with tab1:
    st.subheader("Key Demand Drivers")
    if rev_fi is not None:
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': rev_fi})
        fi_df = fi_df.sort_values(by='Importance', ascending=True).tail(10)
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title='Top 10 Features Driving Course Revenue', color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for the selected model.")

with tab2:
    st.subheader("Category-Level Demand")
    cat_agg = df.groupby('category').agg({'enrollment_count': 'sum', 'course_revenue': 'sum'}).reset_index()
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(cat_agg, values='enrollment_count', names='category', title='Total Enrollments by Category', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(cat_agg, x='category', y='course_revenue', title='Total Revenue by Category', color='category', color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Revenue Distribution by Price Band")
    fig3 = px.box(df, x='price_band', y='course_revenue', color='price_band', title='Revenue Spread per Price Band')
    st.plotly_chart(fig3, use_container_width=True)
