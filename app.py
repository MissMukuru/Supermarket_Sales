import pandas as pd
import plotly.express as px
import streamlit as st

# Set page configuration
st.set_page_config(page_title='Sales dashboard',
                   page_icon=':bar_chart:',
                   layout='wide')

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Sales.csv')
    # Convert 'Time' to hour with coercion for invalid parsing
    df['hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
    
    # Convert 'Date' to datetime with coercion, drop NaT dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    
    # Drop rows with NaN in 'hour' and convert hour to int
    df.dropna(subset=['hour'], inplace=True)
    df['hour'] = df['hour'].astype(int)
    
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Please filter here: ")
city = st.sidebar.multiselect(
    "Select the city: ",
    options=df['City'].unique(),
    default=df['City'].unique()
)
gender = st.sidebar.multiselect(
    "Select the gender: ",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)
customer_type = st.sidebar.multiselect(
    "Select the customer type: ",
    options=df['Customer_type'].unique(),
    default=df['Customer_type'].unique()
)

product_line = st.sidebar.multiselect(
    'Select the product you want to view ',
    options=['All'] + list(df['Product line'].unique()),
    default=['All']
)

# Date filter defaults as python dates for Streamlit
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

# Date input returns a tuple (start_date, end_date)
start_date, end_date = st.sidebar.date_input(
    label='Select the date range you want to view',
    value=(min_date, max_date)
)

# Convert to pd.Timestamp for safe comparison
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

hour_range = st.sidebar.slider(
    'Select Hour Range',
    min_value=int(df['hour'].min()),
    max_value=int(df['hour'].max()),
    value=(int(df['hour'].min()), int(df['hour'].max()))
)

# ---- FIXED FILTERING HERE ----
df_selection = df[
    (df['City'].isin(city)) &
    (df['Gender'].isin(gender)) &
    (df['Customer_type'].isin(customer_type)) &
    (df['Date'] >= start_date) &
    (df['Date'] <= end_date) &
    (df['hour'] >= hour_range[0]) &
    (df['hour'] <= hour_range[1])
]

# Filter product line if not 'All'
if 'All' not in product_line:
    df_selection = df_selection[df_selection['Product line'].isin(product_line)]

# Show filtered data option
if st.sidebar.checkbox('Show Raw Data'):
    st.subheader('Raw filtered Data')
    st.dataframe(df_selection)

# KPIs
total_sales = df_selection['Total'].sum()
avg_rating = round(df_selection['Rating'].mean(), 1)
star_rating = ":star:" * int(round(avg_rating))
avg_sale = round(df_selection['Total'].mean(), 2)

left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Total Sales:")
    st.subheader(f"US $ {total_sales:,.2f}")
with middle_column:
    st.subheader("Average Rating:")
    st.subheader(f"{avg_rating} {star_rating}")
with right_column:
    st.subheader("Average Sale per Transaction:")
    st.subheader(f"US $ {avg_sale}")

st.markdown("---")

# Grouped statistics
group_gender_payment = df_selection.groupby(['Gender', 'Payment'])['Total'].agg(['count', 'sum', 'mean', 'std']).reset_index()
group_branch_product = df_selection.groupby(['Branch', 'Product line'])['Total'].agg(['count', 'sum', 'mean', 'std']).reset_index()
group_city_customer = df_selection.groupby(['City', 'Customer_type'])['Total'].agg(['count', 'sum', 'mean', 'std']).reset_index()
group_hour = df_selection.groupby('hour')['Total'].sum().reset_index()
group_product = df_selection.groupby('Product line')['Total'].agg(['count', 'sum', 'mean', 'std']).reset_index()
group_by_payment_methods = df_selection.groupby('Payment')['Total'].agg(['count', 'sum', 'mean', 'std']).reset_index()
group_gender_quantity = df_selection.groupby('Gender')['Quantity'].agg(['count', 'sum', 'mean', 'std']).reset_index()
group_product_line_quantity = df_selection.groupby('Product line')['Quantity'].sum().reset_index()
group_rating_per_products = df_selection.groupby('Product line')['Rating'].mean().reset_index()
group_taxes_per_gender = df_selection.groupby('Gender')['Tax 5%'].mean().reset_index()
group_taxes_per_product_line = df_selection.groupby('Product line')['Tax 5%'].mean().reset_index()
group_products_per_gross_income = df_selection.groupby('Product line')['gross income'].mean().reset_index()

# Tables
st.subheader('Grouped Statistics')
st.dataframe(group_gender_payment)
st.dataframe(group_branch_product)
st.dataframe(group_city_customer)
st.dataframe(group_hour)
st.dataframe(group_product)
st.dataframe(group_by_payment_methods)
st.dataframe(group_gender_quantity)
st.dataframe(group_product_line_quantity)
st.dataframe(group_rating_per_products)
st.dataframe(group_taxes_per_gender)
st.dataframe(group_taxes_per_product_line)
st.dataframe(group_products_per_gross_income)

st.markdown('---')

# Bar chart: Sales by Gender & Payment
st.subheader("Sales by Gender & Payment Method")
fig1 = px.bar(group_gender_payment,
              x="Gender",
              y="sum",
              color="Payment",
              barmode="group",
              title="Total Sales by Gender and Payment",
              labels={"sum": "Total Sales"},
              template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Branch vs Product Line Sales")
fig2 = px.bar(group_branch_product,
              x="Branch",
              y="sum",
              color="Product line",
              title="Branch-wise Product Line Sales",
              labels={"sum": "Total Sales"},
              template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("City & Customer Type Sales")
fig3 = px.bar(group_city_customer,
              x="City",
              y="sum",
              color="Customer_type",
              barmode="group",
              title="City-wise Sales by Customer Type",
              labels={"sum": "Total Sales"},
              template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Hourly Sales Trend")
fig4 = px.line(group_hour,
               x="hour",
               y="Total",
               markers=True,
               title="Sales by Hour of Day",
               labels={"Total": "Total Sales", "hour": "Hour"},
               template="plotly_white")
st.plotly_chart(fig4, use_container_width=True)

st.subheader("Sales per Product Line")
fig5 = px.bar(group_product,
              x="Product line",
              y="sum",
              title="Total Sales per Product Line",
              labels={"sum": "Total Sales"},
              template="plotly_white")
st.plotly_chart(fig5, use_container_width=True)

st.subheader("Quantity by Gender")
fig6 = px.pie(group_gender_quantity,
              names="Gender",
              values="sum",
              title="Quantity Sold by Gender",
              template="plotly_white")
st.plotly_chart(fig6, use_container_width=True)

st.subheader("Quantity by Product Line")
fig7 = px.bar(group_product_line_quantity,
              x="Product line",
              y="Quantity",
              title="Total Quantity Sold per Product Line",
              labels={"Quantity": "Quantity"},
              template="plotly_white")
st.plotly_chart(fig7, use_container_width=True)

st.subheader("Average Rating by Product Line")
fig8 = px.bar(group_rating_per_products,
              x="Product line",
              y="Rating",
              title="Average Rating per Product Line",
              labels={"Rating": "Average Rating"},
              template="plotly_white")
st.plotly_chart(fig8, use_container_width=True)

st.subheader("Average Tax by Gender")
fig9 = px.bar(group_taxes_per_gender,
              x="Gender",
              y="Tax 5%",
              title="Average Tax Paid by Gender",
              labels={"Tax 5%": "Avg Tax"},
              template="plotly_white")
st.plotly_chart(fig9, use_container_width=True)

st.subheader("Average Tax by Product Line")
fig10 = px.bar(group_taxes_per_product_line,
               x="Product line", y="Tax 5%",
               title="Average Tax Paid by Product Line",
               labels={"Tax 5%": "Avg Tax"},
               template="plotly_white")
st.plotly_chart(fig10, use_container_width=True)

st.subheader("Average Gross Income by Product Line")
fig11 = px.bar(group_products_per_gross_income,
               x="Product line", y="gross income",
               title="Avg Gross Income per Product Line",
               labels={"gross income": "Gross Income"},
               template="plotly_white")
st.plotly_chart(fig11, use_container_width=True)
