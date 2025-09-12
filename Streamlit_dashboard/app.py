import pandas as pd
import plotly.express as ps
import streamlit as st

st.set_page_config(page_title = 'Sales dashboard',
                    page_icon = ':bar_chart',
                    layout = 'wide')

df = pd.read_excel(
    io = 'supermarkt_sales.xlsx',
    engine = 'openpyxl',
    sheet_name= 'Sales',
    skiprows=3,
    usecols='B:R',
    nrows = 1000
)

st.dataframe(df)

st.sidebar.header("Please filter here: ")
city = st.sidebar.multiselect(
    "Select the city: ",
    options = df['City'].unique(),
    default = df['City'].unique()
)

gender = st.sidebar.multiselect(
    "Select the city: ",
    options = df['Gender'].unique(),
    default = df['Gender'].unique()
)

customer_type = st.sidebar.multiselect(
    "Select the city: ",
    options = df['Customer_type'].unique(),
    default = df['Customer_type'].unique()
)

df.