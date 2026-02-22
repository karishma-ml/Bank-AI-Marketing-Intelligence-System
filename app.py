import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# PAGE CONFIG
st.set_page_config(
    page_title="FinBank AI Marketing Intelligence",
    page_icon="🏦",
    layout="wide"
)
corpus = joblib.load("corpus.pkl")

# LOAD CSS
# -------------------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -------------------------
# SIMPLE AUTHENTICATION
# -------------------------

credentials = {
    "director": "dir123",
    "trainer": "tra123",
    "manager": "man123"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 FinBank Secure Login")
    st.markdown("### Authorized Banking Personnel Only")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in credentials and credentials[username] == password:
            st.session_state.logged_in = True
            st.success(f"Welcome {username}")
            st.rerun()
        else:
            st.error("Invalid Credentials")

if not st.session_state.logged_in:
    login_page()
    st.stop()

# LOAD DATA CORRECTLY
try:
    df = pd.read_csv("bank_data.csv",sep=";",engine="python")

    # If still only one column, manually split
    if len(df.columns) == 1:
        df = df[df.columns[0]].str.split(";", expand=True)

        # Set correct column names manually
        df.columns = [
            "age","job","marital","education","default",
            "housing","loan","contact","month","day_of_week",
            "duration","campaign","pdays","previous","poutcome",
            "emp.var.rate","cons.price.idx","cons.conf.idx",
            "euribor3m","nr.employed","y"
        ]

    # # Clean column names
    df.columns = df.columns.str.strip().str.lower()

except Exception as e:
    st.error(f"Dataset Error: {e}")
    st.stop()

# SIDEBAR
st.sidebar.title("🏦 FinBank AI System")
menu = st.sidebar.radio("Navigation", [
    "Executive Dashboard",
    "Dataset Tools",
    "Subscription Prediction",
    "Campaign Insights",
    "AI Boot",
    "About Project"
])

# EXECUTIVE DASHBOARD
if menu == "Executive Dashboard":

    st.title("Executive Marketing Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    if "y" in df.columns:
        subscribed = df["y"].value_counts().get("yes", 0)
        not_subscribed = df["y"].value_counts().get("no", 0)
    else:
        subscribed = 0
        not_subscribed = 0
        st.warning("Target column 'y' not found in dataset")

    col2.metric("Subscribed Clients", subscribed)
    col3.metric("Non-Subscribed", not_subscribed)

    st.markdown("""
    ### Business Objective

    This AI system analyzes marketing campaign data to predict
    customer subscription behavior.

    Key variables include:
    - Demographics (age, job, marital, education)
    - Campaign metrics (duration, contacts)
    - Economic indicators (euribor3m, employment rate)
    """)

# DATASET TOOLS
elif menu == "Dataset Tools":

    st.title("📂 Dataset Utilities")

    section = st.sidebar.selectbox(
        "Select Dataset Option",
        ["Dataset Preview", "Dataset Information", "Numerical Summary"])

    # -------- Dataset Preview --------
    if section == "Dataset Preview":

        view_options = st.sidebar.radio( "Select to View Dataset",["Show", "Hide"])

        if view_options == "Show":
            st.subheader("✨ Dataset Preview")
            st.dataframe(df.head())

    # -------- Dataset Info --------
    elif section == "Dataset Information":

        st.subheader("🎉 Dataset Information")

        col1, col2 = st.columns(2)
        col1.metric("Number of Rows", df.shape[0])
        col2.metric("Number of Columns", df.shape[1])

    # -------- Numerical Summary --------
    elif section == "Numerical Summary":
        with st.expander("📊 Summary of Numerical Columns"):
            st.write(df.describe())

# SUBSCRIPTION PREDICTION
elif menu == "Subscription Prediction":

    st.title("Customer Subscription Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Enter your age", min_value=20, max_value=90)
        duration = st.number_input("Last contact duration")
        campaign = st.number_input("Number of contacts during this campaign")
        pdays = st.number_input("Days since client was last contacted")
        previous = st.number_input("Number of contacts before this campaign")
        emp_var_rate = st.number_input("Employment variation rate - quarterly")
        cons_price_idx = st.number_input("Consumer price index - monthly")
        cons_conf_idx = st.number_input("Consumer confidence index - monthly")
        euribor3m = st.number_input("Euribor 3-month rate - daily")
        nr_employed = st.number_input("Number of employees - quarterly")

    with col2:
        job = st.selectbox("Job of client", ["admin", "blue-collar", "technician", "services", "management","retired", "entrepreneur", "self-employed", "housemaid", "unemployed", "student"])
        marital = st.selectbox("Marital status of the client", ['married', 'single', 'divorced'])
        education = st.selectbox("Highest level of education", ['university.degree', 'high.school', 'basic.9y', 'professional.course','basic.4y', 'basic.6y', 'illiterate'])
        default = st.selectbox("Has credit in default", ['no', 'yes'])
        housing = st.selectbox("Has a housing loan?", ['yes', 'no'])
        loan = st.selectbox("Did you take any loan?", ['no', 'yes'])
        contact = st.selectbox("Select your contact mode", ['cellular', 'telephone'])
        month = st.selectbox("Select a month", ['may', 'jul', 'aug', 'jun', 'nov', 'apr', 'oct', 'sep', 'mar', 'dec'])
        day = st.selectbox("Select a day", ['thu', 'mon', 'wed', 'tue', 'fri'])
        poutcome = st.selectbox("Select previous marketing campaign", ['nonexistent', 'failure', 'success'])

        # Define categorical features
    cat_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

    # Build a dictionary for features (easy to debug & visualize)
    input_dict = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed
    }

    from catboost import Pool, CatBoostClassifier
    cbc= CatBoostClassifier()

    # Load trained CatBoost model
    cbc = joblib.load("cb.pkl")

    # Convert to DataFrame-like structure for CatBoost
    input_df = pd.DataFrame([input_dict])

    # Build CatBoost test pool
    test_pool = Pool(data=input_df, cat_features=cat_features, feature_names= input_df.columns.tolist())

    if st.button("😀 Predict"):
        pred= cbc.predict(test_pool)[0]
        st.session_state["pred"]= pred 
        st.success(f"Predictions: {pred}")
        probability = cbc.predict_proba(input_df)[0][1]

        if pred == "yes":
            st.success(f"High Subscription Probability: {probability:.2%}")
        else:
            st.error(f"Low Subscription Probability: {probability:.2%}")


# CAMPAIGN INSIGHTS
elif menu == "Campaign Insights":

    st.title("Campaign Analytics")

    fig1 = px.histogram(df,x="age",nbins=30,title="Customer Age Distribution",
        labels={"age": "Customer Age","count": "Number of Customers"})

    # Transparent + professional style
    fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",title_x=0.5,font=dict(size=14),xaxis=dict(title="Customer Age"),
        yaxis=dict(title="Number of Customers"),)

    fig1.update_traces(marker_color="#1D70B8",opacity=0.85)

    st.plotly_chart(fig1, use_container_width=True)

    # ---------- CALL DURATION VS SUBSCRIPTION ----------
    st.subheader("📊 Data Analysis Insights")

# ---------- Q1 ----------
    with st.expander("Q1: What is the age distribution of customers?"):

       st.write("Majority of clients fall between 30–50 years age group.")

       fig_age = px.histogram(df,x="age",nbins=20,title="Customer Age Distribution",labels={"age": "Customer Age","count": "Number of Customers"})
       fig_age.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",title_x=0.5)
       fig_age.update_traces(marker_color="#1D70B8")

       st.plotly_chart(fig_age, use_container_width=True)

# ---------- Q2 ----------
    with st.expander("Q2: What is the distribution of customers by job type?"):

       job_counts = df["job"].value_counts().reset_index()
       job_counts.columns = ["Job", "Count"]
       fig_job = px.pie(job_counts,names="Job",values="Count",title="Customer Distribution by Job Type")
       fig_job.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",title_x=0.5)

       st.plotly_chart(fig_job, use_container_width=True)

# ---------- Q3 ----------
    with st.expander("Q3: What is the distribution of customers by education level?"):

        edu_counts = df["education"].value_counts().reset_index()
        edu_counts.columns = ["Education Level", "Count"]

        fig_edu = px.bar(edu_counts,x="Education Level",y="Count",title="Customer Distribution by Education Level",
        labels={"Education Level": "Education Level","Count": "Number of Customers"})
        fig_edu.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",title_x=0.5)
        fig_edu.update_traces(marker_color="#D4AF37")

    st.plotly_chart(fig_edu, use_container_width=True)


# ---------- Q4 ----------
    with st.expander("Q4: What is the outcome distribution of previous marketing campaigns?"):

       poutcome_counts = df["poutcome"].value_counts().reset_index()
       poutcome_counts.columns = ["Previous Campaign Outcome", "Count"]

       fig_camp = px.pie(poutcome_counts,names="Previous Campaign Outcome",values="Count",title="Previous Marketing Campaign Outcomes")
       fig_camp.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",title_x=0.5)

    st.plotly_chart(fig_camp, use_container_width=True)

# Chatbot
elif menu == "AI Boot":

    st.title("🤖 FinBank AI Assistant")
    # Initialize chat history
    if "chat_history" not in st.session_state:
      st.session_state.chat_history = []

# Chatbot function
    def chatbot_response(user_text):
        text = user_text.lower()
        for q, a in corpus.items():
            words = q.lower().split()[:2]
            if all(w in text for w in words):
                return a
        return corpus.get("default", "Sorry, I don't understand.")

    user_input = st.text_input("Ask about Bank Dataset ...")

    if user_input:
        reply = chatbot_response(user_input)
        st.session_state.chat_history.append(("You: " + user_input, "Bot: " + reply))

    for u, b in st.session_state.chat_history:
        st.write(u)
        st.write(b)
        

# ABOUT PROJECT
elif menu == "About Project":

    st.title("FinBank AI Marketing Intelligence System")

    st.markdown("""
    ### Project Overview

    This project uses the Bank Marketing Dataset to build a
    machine learning system that predicts whether a client
    will subscribe to a term deposit.

    ### Dataset Contains:
    - Customer Demographics (age, job, marital, education)
    - Financial Attributes (loan, housing, default)
    - Campaign Information (duration, contacts, previous outcome)
    - Economic Indicators (euribor3m, employment rate)

    ### Model Used:
    CatBoost Classifier (Binary Classification)

    ### Business Value:
    - Smarter campaign targeting
    - Reduced marketing cost
    - Increased ROI
    - Better customer segmentation

    ⚠ This system is for educational demonstration only.
    """)
    st.markdown("""
### Developed By
**Karishma** 
""")
