# app.py
import streamlit as st
import pandas as pd
import ml_module as ml
import numpy as np
import pickle
import joblib
import procedure_process_predict as ppp
import sqlite3
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from pathlib import Path
import calendar
import os

# ================= Safe App Startup (Jupyter + Script) =================
import os
import sys
from pathlib import Path

# Determine project root safely
try:
    # If running as a script
    PROJECT_ROOT = Path(__file__).parent.resolve()
except NameError:
    # If running in Jupyter, use current working directory
    PROJECT_ROOT = Path.cwd().resolve()

# Change working directory to project root
os.chdir(PROJECT_ROOT)

# Define folders relative to project root
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

# Optional: Filter stderr to hide absolute paths (like .venv310 full path)
class SafeLogger:
    def write(self, msg):
        venv_path = str((PROJECT_ROOT / ".venv310").resolve())
        msg_safe = msg.replace(venv_path, ".venv310")
        sys.__stderr__.write(msg_safe)
    def flush(self):
        sys.__stderr__.flush()

sys.stderr = SafeLogger()

# Suppress tqdm notebook warnings globally
import warnings
from tqdm import TqdmWarning
warnings.filterwarnings("ignore", category=TqdmWarning)

# Optional: safe tqdm wrapper
from tqdm import tqdm

def in_jupyter():
    try:
        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell", "Shell")
    except NameError:
        return False

def safe_tqdm(iterable, **kwargs):
    if in_jupyter():
        return tqdm(iterable, notebook=False, **kwargs)
    else:
        return tqdm(iterable, **kwargs)

# ================= End of Safe Startup =================



# Base directory where app.py is located
#BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path.cwd()  # Jupyter-safe
MODELS_DIR = BASE_DIR / "models" 

# Safe path to the CSV file
DATA_DIR = BASE_DIR / "data"  # storing CSV in a data/ folder
file_path = DATA_DIR / "healthcare-procedure-cost-dataset.csv"

# Safe paths to the pickle files
scaler_path = BASE_DIR / 'models' / 'scaler.pkl'
selector_path = BASE_DIR / 'models' / 'selector.pkl'
imputer_path = BASE_DIR / 'models' / 'imputer.pkl'
model_path = BASE_DIR / 'models' / 'model.pkl'

# Safe path to the database
DB_PATH = BASE_DIR / "data" / "procedure_costs_database.db"  # storing DB in data/

def load_model():
    # Use joblib to load the model if it was saved using joblib
    model_path = BASE_DIR / 'models' / 'model.pkl'
    model = joblib.load(model_path)
    return model
    
model = load_model()

# Loading original data
df = pd.read_csv(file_path)
columns = df.columns.tolist()
print(columns)

# Page configuration
st.set_page_config(
    page_title = "Cost Prediction",
    page_icon = "",
    layout = "centered"
)
st.title("Cost Prediction App and Procedure Cost Database Querying")

demo_number = st.slider("Choose a demo number", min_value = 1, max_value = 10, value =1)
demo_number_index = demo_number - 1

# Preparing a demo input for prediction
df_input_all = df.copy()
df_input = df_input_all.iloc[[demo_number_index]]
df_input_reshape = df_input.values.reshape(1,-1)
X_test_selected = ppp.process(df_input)

# Get the selected features from the selector object (has top 15)

selector = joblib.load(selector_path)  # Load the selector object

# Original column names, replace this with the actual feature names from your dataset
selected_column_names = ['patient_age',
                'patient_gender',
                'patient_race_ethnicity',
                'insurance_type',
                'procedure_description',
                'cost_paid',
                'cost_patient_responsibility',
                'cost_insurance_covered',
                'procedure_outcome',
                'procedure_date_year',
                'procedure_date_month',
                'AgeGroup_Teen',
                'AgeGroup_Adult',
                'AgeGroup_Middle',
                'AgeGroup_Senior'
                ]

X_test_selected = pd.DataFrame(X_test_selected, columns = selected_column_names)

# Display the row of the dataset for demo
st.write("Original Input Data")
example_row_df = pd.DataFrame(df_input_reshape, columns = df.columns)
# dimensions checks
print(type(df_input_reshape))
print(df_input_reshape.shape)
st.write(example_row_df)

st.write("Application - Predict the Billed Cost based on given parameters:")
st.write(X_test_selected)  # Display the first row as a preview

#dataframe version
input = X_test_selected.iloc[0].values  # Convert the first row to a NumPy array

# Reshape the first row to be a 2D array (1 sample, multiple features)
input_reshaped = input.reshape(1, -1)


if "predicted_cost" not in st.session_state:
    st.session_state.predicted_cost = None

if st.button("Predict Cost"):
    st.session_state.predicted_cost = ml.predict(model, input_reshaped)

if st.session_state.predicted_cost is not None:
    st.success(
        f"Predicted cost: ${st.session_state.predicted_cost[0]:,.2f}"
    )

################ 
# Database Data
################
# Function to create a connection to the SQLite database
def create_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

# Function to fetch data from the database
@st.cache
def fetch_data(query):
    conn = create_connection()  # Ensure connection is created inside the function
    df = pd.read_sql_query(query, conn)
    conn.close()  # Always close the connection after use
    return df


# WIth filters
# Sidebar for navigation
st.sidebar.title("Streamlit App Sections")
section = st.sidebar.radio("Select Section", ["Table Overview", "Plots", "Statistics", "Research Query"])

# Main content
if section == "Statistics":
    st.header("Summary Statistics")

    # Procedure Type Filter (Dropdown)
    st.subheader("Filter by Procedure Type")
    query_procedure = "SELECT DISTINCT procedure_description FROM procedures"
    procedures_df = fetch_data(query_procedure)
    procedure_types = procedures_df['procedure_description'].unique()
    selected_procedure = st.selectbox('Select Procedure Type', procedure_types)

    # Display statistics for the selected procedure type
    st.write(f"Statistics for Procedure: {selected_procedure}")
    query_procedure_stats = f"""
        SELECT procedure_description, COUNT(cost_id) AS num_records, AVG(cost_billed) AS avg_billed, 
               AVG(cost_paid) AS avg_paid, MAX(cost_billed) AS max_billed, MIN(cost_billed) AS min_billed
        FROM costs 
        JOIN procedures ON costs.procedure_id = procedures.procedure_id
        WHERE procedure_description = '{selected_procedure}'
        GROUP BY procedure_description
    """
    procedure_stats_df = fetch_data(query_procedure_stats)
    st.dataframe(procedure_stats_df)

    # Age Category Filter for Costs (Dropdown)
    st.subheader("Filter by Age Category")
    age_categories = ['Child', 'Teen', 'Adult', 'Middle', 'Senior']
    selected_age_category = st.selectbox('Select Age Category', age_categories)

    # Display statistics for the selected age category
    st.write(f"Statistics for Age Category: {selected_age_category}")
    query_age_category_stats = f"""
        SELECT patient_age_group, COUNT(cost_id) AS num_records, AVG(cost_billed) AS avg_billed, 
               AVG(cost_paid) AS avg_paid, MAX(cost_billed) AS max_billed, MIN(cost_billed) AS min_billed
        FROM costs 
        JOIN patients ON costs.patient_id = patients.patient_id
        WHERE patient_age_group = '{selected_age_category}'
        GROUP BY patient_age_group
    """
    age_category_stats_df = fetch_data(query_age_category_stats)
    st.dataframe(age_category_stats_df)

elif section == "Table Overview":
    st.header("Database Tables")

    # Display Procedures Table
    st.subheader("Procedures Table")
    query = "SELECT * FROM procedures"
    procedures_df = fetch_data(query)
    st.dataframe(procedures_df)

    # Display Patients Table
    st.subheader("Patients Table")
    query = "SELECT * FROM patients"
    patients_df = fetch_data(query)
    st.dataframe(patients_df)

    # Display Providers Table
    st.subheader("Providers Table")
    query = "SELECT * FROM providers"
    providers_df = fetch_data(query)
    st.dataframe(providers_df)

    # Display Costs Table
    st.subheader("Costs Table")
    query = "SELECT * FROM costs"
    costs_df = fetch_data(query)
    st.dataframe(costs_df)

elif section == "Plots":
    st.header("Visualizations")

    # Plot 0. Plotting the Costs for Women vs Men by Age Group
    st.subheader("Costs for Women vs Men by Age Group")

    # Query to get the patient data with age, gender, and costs
    query = """
    SELECT p.patient_age, p.patient_gender, c.cost_billed
    FROM patients p
    JOIN costs c ON p.patient_id = c.patient_id
    """
    df = fetch_data(query)

    # Convert patient_age to numeric (in case it is a string)
    df['patient_age'] = pd.to_numeric(df['patient_age'], errors='coerce')

    # Handle any rows where conversion failed (NaN values)
    df = df.dropna(subset=['patient_age', 'cost_billed'])  # Remove rows with NaN values

    # Create Age Groups (bins)
    age_bins = [0, 12, 18, 35, 60, 100]
    age_labels = ['Child', 'Teen', 'Adult', 'Middle', 'Senior']
    df['AgeGroup'] = pd.cut(df['patient_age'], bins=age_bins, labels=age_labels, right=False, ordered = True)

    # Group by Gender and AgeGroup, then calculate total cost billed
    gender_age_grouped = df.groupby(['AgeGroup', 'patient_gender']).agg(total_cost=('cost_billed', 'sum')).reset_index()

    # Check if we have data for both genders
    if gender_age_grouped['patient_gender'].nunique() != 2:
        st.warning("Gender data seems to have non-standard values. Only 'Male' and 'Female' should be present.")

    # Pivot the data so that we can compare costs by gender and age group
    gender_age_pivot = gender_age_grouped.pivot(index='AgeGroup', columns='patient_gender', values='total_cost')

    # Plotting the bar chart
    fig = px.bar(
        gender_age_grouped,
        x='AgeGroup',
        y='total_cost',
        color='patient_gender',
        barmode='group',
        text='total_cost',  # optional: display values on bars
        title='Total Cost by Gender and Age Group',
        labels={'total_cost': 'Total Cost (USD)', 'AgeGroup': 'Age Group', 'patient_gender': 'Gender'},
        color_discrete_map={
            'Male': 'blue',
            'Female': 'orange',
            'Other': 'green',
            'Unknown': 'purple'
        }
    )
    
    # Rotate x-axis labels
    fig.update_layout(
        xaxis_tickangle=45,
        xaxis_categoryorder='array',  # ensures ordered categorical
        xaxis_categoryarray=age_labels,
        legend_title_text='Gender',
        yaxis_tickprefix='$'
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Plot 2: Boxplot of cost paid by procedure outcome
    st.subheader("Cost Paid by Procedure Outcome")
    query = "SELECT procedure_outcome, cost_paid FROM costs"
    costs_df = fetch_data(query)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='procedure_outcome', y='cost_paid', data=costs_df, palette="Set2")
    st.pyplot(plt)


    # Plot 3
    st.subheader("Cost Billed by Procedure Date Month")
    
    # Fetch the aggregated data
    query = """
    SELECT 
        p.procedure_date_month,
        SUM(c.cost_billed) AS total_cost
    FROM procedures p
    JOIN costs c ON p.procedure_id = c.procedure_id
    GROUP BY p.procedure_date_month
    ORDER BY p.procedure_date_month;
    """
    monthly_df = fetch_data(query)
    
    # Ensure month column is datetime for proper chronological ordering
    monthly_df['procedure_date_month'] = monthly_df['procedure_date_month'].apply(lambda x: calendar.month_abbr[x])
    
    # Plot total cost per month
    fig = px.bar(
        monthly_df,
        x='procedure_date_month',
        y='total_cost',
        title="Total Cost Billed per Month",
        labels={'procedure_date_month': 'Month', 'total_cost': 'Total Cost (USD)'},
        text='total_cost'
    )
    
    # Layout adjustments
    fig.update_layout(
        xaxis_tickangle=45,
        yaxis_tickprefix='$'
    )
    
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)



elif section == "Research Query":
    st.header("Custom Research Query")

    # Pre-specified queries dropdown
    st.subheader("Select a pre-specified query")
    queries = {
        "Three most common expensive providers in each US state by total cost billed": """
        WITH ProviderCosts AS (
            SELECT
                pr.provider_state,
                pr.provider_name,
                SUM(c.cost_billed) AS total_cost_billed,
                ROW_NUMBER() OVER (PARTITION BY pr.provider_state ORDER BY SUM(c.cost_billed) DESC) AS provider_rank
            FROM
                costs c
            JOIN
                providers pr ON c.provider_id = pr.provider_id
            GROUP BY
                pr.provider_state, pr.provider_name
        )
        SELECT
            provider_state,
            provider_name,
            total_cost_billed
        FROM
            ProviderCosts
        WHERE
            provider_rank <= 3
        ORDER BY
            provider_state,
            provider_rank;
        """,
        
        "The procedures with the worst outcomes": """
        SELECT p.procedure_code, p.procedure_description,
        c.procedure_outcome,  -- Directly using procedure_outcome from costs table
        COUNT(*) AS outcome_count
        FROM procedures p
        JOIN costs c ON c.procedure_id = p.procedure_id
        WHERE c.procedure_outcome IN ('complications', 'readmission')  -- Filtering on procedure_outcome directly from costs
        GROUP BY p.procedure_code, p.procedure_description, c.procedure_outcome  -- Grouping by both code and description
        ORDER BY outcome_count DESC
        LIMIT 5;
        """,
                   
        "Three most common expensive providers in each US state by total cost billed ": """
        WITH ProviderCosts AS (
            SELECT
                pr.provider_state,
                pr.provider_name,
                SUM(c.cost_billed) AS total_cost_billed,
                ROW_NUMBER() OVER (PARTITION BY pr.provider_state ORDER BY SUM(c.cost_billed) DESC) AS provider_rank
            FROM
                costs c
            JOIN
                providers pr ON c.provider_id = pr.provider_id
            GROUP BY
                pr.provider_state, pr.provider_name
        )
        SELECT
            provider_state,
            provider_name,
            total_cost_billed
        FROM
            ProviderCosts
        WHERE
            provider_rank <= 3
        ORDER BY
            provider_state,
            provider_rank;
        """,
        
        "Three most common procedures in each gender group": """
        WITH RankedProcedures AS (
            SELECT
                p.patient_gender,
                pr.procedure_description,
                COUNT(*) AS procedure_count,
                ROW_NUMBER() OVER (PARTITION BY p.patient_gender ORDER BY COUNT(*) DESC) AS procedure_rank
            FROM
                costs c
            JOIN
                patients p ON c.patient_id = p.patient_id
            JOIN
                procedures pr ON c.procedure_id = pr.procedure_id
            GROUP BY
                p.patient_gender, pr.procedure_description
        )
        SELECT
            patient_gender,
            procedure_description,
            procedure_count
        FROM
            RankedProcedures
        WHERE
            procedure_rank <= 3
        ORDER BY
            patient_gender,
            procedure_rank;
        """,
        
        "Three most common procedures in each age group": """
    
        WITH RankedProcedures AS (
            SELECT
                p.patient_age_group,
                pr.procedure_description,
                COUNT(*) AS procedure_count,
                ROW_NUMBER() OVER (PARTITION BY p.patient_age_group ORDER BY COUNT(*) DESC) AS procedure_rank
            FROM
                costs c
            JOIN
                patients p ON c.patient_id = p.patient_id
            JOIN
                procedures pr ON c.procedure_id = pr.procedure_id
            GROUP BY
                p.patient_age_group, pr.procedure_description
        )
        SELECT
            patient_age_group,
            procedure_description,
            procedure_count
        FROM
            RankedProcedures
        WHERE
            procedure_rank <= 3
        ORDER BY
            patient_age_group,
            procedure_rank;
        
        """
    }

    # Dropdown for selecting query
    selected_query_name = st.selectbox("Choose a research question", list(queries.keys()))
    selected_query = queries[selected_query_name]

    # Run the selected query and display the results
    if st.button("Run Query"):
        try:
            result_df = fetch_data(selected_query)
            st.dataframe(result_df)  # Display the result of the custom query
        except Exception as e:
            st.error(f"Error: {str(e)}")  # If there's an error, display it