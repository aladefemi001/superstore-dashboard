
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("Sample - Superstore.csv", encoding='ISO-8859-1')
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Month'] = df['Order Date'].dt.to_period('M')

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Data")
regions = df['Region'].unique()
segments = df['Segment'].unique()
selected_regions = st.sidebar.multiselect("Select Region(s):", regions, default=regions)
selected_segments = st.sidebar.multiselect("Select Segment(s):", segments, default=segments)
date_range = st.sidebar.date_input("Select Date Range:", [df['Order Date'].min(), df['Order Date'].max()])

# Apply filters
filtered_df = df[
    (df['Region'].isin(selected_regions)) &
    (df['Segment'].isin(selected_segments)) &
    (df['Order Date'] >= pd.to_datetime(date_range[0])) &
    (df['Order Date'] <= pd.to_datetime(date_range[1]))
]

st.title("📊 Executive Dashboard - Superstore Analysis")

# --- Revenue & Profit Heatmap ---
st.header("Profit Heatmap by Region & Segment")
heatmap_data = filtered_df.groupby(['Region', 'Segment'])['Profit'].sum().unstack()
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm", ax=ax1)
st.pyplot(fig1)

# --- Monthly Sales Trends ---
st.header("📈 Monthly Sales Trend")
monthly_sales = filtered_df.groupby('Month')['Sales'].sum().reset_index()
monthly_sales['Month'] = monthly_sales['Month'].astype(str)
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=monthly_sales, x='Month', y='Sales', marker='o', ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)

# --- Customer Churn ---
st.header("🔁 Customer Churn Rate")
churn_cutoff = filtered_df['Order Date'].max() - pd.Timedelta(days=180)
customer_last = filtered_df.groupby('Customer ID')['Order Date'].max().reset_index()
customer_last['Churned'] = customer_last['Order Date'] < churn_cutoff
churn_rate = customer_last['Churned'].mean()
st.metric("Churn Rate", f"{churn_rate:.2%}")

# --- Inventory Turnover ---
st.header("🐢 Slow-Moving Inventory")
turnover = filtered_df.groupby('Product Name').agg({'Quantity': 'sum', 'Order Date': 'nunique'}).reset_index()
turnover['Turnover Ratio'] = turnover['Quantity'] / turnover['Order Date']
slow_movers = turnover.sort_values('Turnover Ratio').head(10)
st.dataframe(slow_movers[['Product Name', 'Turnover Ratio']])

# --- High-Impact Recommendations ---
st.header("💡 Recommendations")
loss_making = filtered_df.groupby('Sub-Category')[['Sales', 'Profit']].sum().reset_index()
loss_making = loss_making[loss_making['Profit'] < 0].sort_values('Profit')
loss_subcats = loss_making.head(3)['Sub-Category'].tolist()

st.markdown(f"""
1. ❌ Eliminate loss-making sub-categories: **{', '.join(loss_subcats)}**  
2. 📈 Focus on high-LTV customer segments in profitable regions.  
3. 🔻 Review over-discounting in low-profit areas.  
""")

# --- Optional: Churn Prediction Model ---
st.header("🧠 Customer Churn Prediction (Simulated)")
rfm = filtered_df.groupby('Customer ID').agg({
    'Order Date': [lambda x: (x.max() - x.min()).days, 'count'],
    'Sales': 'sum'
}).reset_index()
rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
rfm['Churned'] = rfm['Recency'] > 180

X = rfm[['Recency', 'Frequency', 'Monetary']]
y = rfm['Churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
