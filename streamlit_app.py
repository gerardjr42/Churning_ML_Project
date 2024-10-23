import streamlit as st
import utils as ut
import pandas as pd
import numpy as np
import pickle
import os
from openai import OpenAI

# Try using groq_api_key if it exists
if 'GROQ_API_KEY' in os.environ:
    api_key = os.environ['GROQ_API_KEY']
else:
    # Fallback to secretName if groq_api_key is not found
    api_key = st.secrets["GROQ_API_KEY"]

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

@st.cache_data
def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

# Load Models
xgboost_model = load_model('xgb_model.pkl')
naive_bayes_model = load_model('nb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
svm_model = load_model('svm_model.pkl')
knn_model = load_model('knn_model.pkl')
voting_classifier_model = load_model('voting_clf.pkl')
xgboost_SMOTE_model = load_model('xgboost-SMOTE.pkl')
xgboost_featureEngineered_model = load_model('xgboost-featureEngineered.pkl')

def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == "France" else 0,
        'Geography_Germany': 1 if location == "Germany" else 0,
        'Geography_Spain': 1 if location == "Spain" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
    }
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

def make_predictions(input_df, input_dict):
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'Support Vector Machine': svm_model.predict_proba(input_df)[0][1],
        # 'K Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
        # 'Naive Bayes': naive_bayes_model.predict_proba(input_df)[0][1],
        # 'Decision Tree': decision_tree_model.predict_proba(input_df)[0][1],
    }

    avg_probability = np.mean(list(probabilities.values()))

    # Add some vertical space
    st.markdown("<div style='margin-top: 60px;'></div>", unsafe_allow_html=True)

    # Create a 2-column layout with added spacing
    col1, spacer, col2 = st.columns([1, 0.1, 1])

    with col1:
        st.markdown("<h4 style='text-align: center;'>Churn Probability</h4>", unsafe_allow_html=True)
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        st.markdown("<h4 style='text-align: center;'>Churn Probability by Model</h4>", unsafe_allow_html=True)
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True, config={'displayModeBar': False})

    st.markdown(f"<p style='text-align: center;'>The customer has a {avg_probability:.2%} probability of churning.</p>", unsafe_allow_html=True)

    # Add some vertical space
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

    return avg_probability

def generate_percentiles(df, input_dict):
    all_num_products = df['NumOfProducts'].sort_values().tolist()
    all_balances = df['Balance'].sort_values().tolist()
    all_estimated_salaries = df['EstimatedSalary'].sort_values().tolist()
    all_tenures = df['Tenure'].sort_values().tolist()
    all_credit_scores = df['CreditScore'].sort_values().tolist()
    product_rank = np.searchsorted(all_num_products, input_dict['NumOfProducts'], side='right')
    balance_rank = np.searchsorted(all_balances, input_dict['Balance'], side='right')
    salary_rank = np.searchsorted(all_estimated_salaries, input_dict['EstimatedSalary'], side='right')
    tenure_rank = np.searchsorted(all_tenures, input_dict['Tenure'], side='right')
    credit_rank = np.searchsorted(all_credit_scores, input_dict['CreditScore'], side='right')

    N = 10000

    percentiles = {
        'CreditScore': int(np.floor((credit_rank / N) * 100)),
        'Tenure': int(np.floor((tenure_rank / N) * 100)),
        'EstimatedSalary': int(np.floor((salary_rank / N) * 100)),
        'Balance': int(np.floor((balance_rank / N) * 100)),
        'NumOfProducts': int(np.floor((product_rank / N) * 100)),
    }
    return percentiles

def explain_prediction(probability, input_dict, surname):
    prompt = f"""You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.

    Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning, based on the information provided below.

    Here is the customer's information:
    {input_dict}

    Here are the machine learning model's top 10 most important features for predicting churn:

                Feature  | Importance
    -----------------------------------
          NumOfProducts  |	0.23
                    Age  |	0.22
         IsActiveMember  |  0.13
      Geography_Germany  |	0.12
                Balance  | 	0.08
                 Gender  | 	0.08
       Geography_France  | 	0.06
              HasCrCard  | 	0.02
            CreditScore  |	0.02
        EstimatedSalary  | 	0.02
                 Tenure  | 	0.02
        Geography_Spain  | 	0.00

    {pd.set_option('display.max_columns', None)}

    Here are summary statistics for churned customers:
    {df[df['Exited'] == 1].describe()}

    Here are summary statistics for non-churned customers:
    {df[df['Exited'] == 0].describe()}

    - If the customer has over a 40% risk of churning, generate a 3 sentence explanation of why they are at risk of churning.
    - Else, if the customer has less than a 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk of churning.
    - Your explanation should be based on your analysis, the summary statistics of churned and non-churned customers, and the feature importance provided.
    """
    print("EXPLANATION PROMPT", prompt)

    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )
    return raw_response.choices[0].message.content

def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""You are a manager at HS Bank. You are responsible for ensuring customers stay with the bank and are incentivized with various offers.

    You noticed a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.

    Here is the customer's information:
    {input_dict}

    Here is some explanation as to why the customer might be at risk of churning: {explanation}

    Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyal to the bank.

    Make sure to list out a set of incentives to stay based on their information, in bullet point format. Don't ever mention the probability of churning, or the machine learning model to the customer. Check grammar and sound professional. Do not include phone numbers or email addresses.
    """

    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )

    print("\n\nEMAIL PROMPT", prompt)
    return raw_response.choices[0].message.content

st.title("Customer Churn Prediction")

@st.cache_data
def load_data():
    return pd.read_csv("churn.csv")

df = load_data()

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_surname = selected_customer_option.split(" - ")[1]
    selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=int(selected_customer['CreditScore']))

        location = st.selectbox(
            "Location", ["Spain", "France", "Germany"],
            index=["Spain", "France", "Germany"].index(
                selected_customer['Geography']))

        gender = st.radio("Gender", ["Male", "Female"],
                          index=0 if selected_customer['Gender'] == "Male" else 1)
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=int(selected_customer["Age"]))

        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=50,
            value=int(selected_customer["Tenure"]))

    with col2:
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=float(selected_customer['Balance']))

        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=int(selected_customer['NumOfProducts']))

        has_credit_card = st.checkbox(
            "Has Credit Card",
            value=bool(selected_customer['HasCrCard']))

        is_active_member = st.checkbox(
            "Is Active Member",
            value=bool(selected_customer['IsActiveMember']))

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer['EstimatedSalary']))

    input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)

    avg_probability = make_predictions(input_df, input_dict)

    percentiles = generate_percentiles(df, input_dict)
    fig = ut.create_percentile_chart(percentiles)
    st.plotly_chart(fig, use_container_width=True)

    explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])

    st.markdown("---")
    st.header("Explanation of Prediction")
    st.markdown(explanation)

    email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])

    st.markdown("---")
    st.subheader("Personalized Email")
    st.markdown(email)
