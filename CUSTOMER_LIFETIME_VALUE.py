############################################
# CUSTOMER LIFETIME VALUE
############################################

############################################
#FIRST APPROACH
############################################

# 1. Data Preparation
# 2. Average Order Value (average_order_value = total_price / total_transaction)
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
# 4. Repeat Rate & Churn Rate (number of customers who made multiple purchases / all customers)
# 5. Profit Margin (profit_margin =  total_price * 0.10)
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
# 8. Creating Segments

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
import researchpy as rp
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
from sklearn.preprocessing import MinMaxScaler
#pd.set_option('display.float_format', lambda x: '%.4f' % x)

df = pd.read_excel("/Users/serhandulger/PycharmProjects/DSMLBC_7/WEEK_3/DATASETS/online_retail_II.xlsx", sheet_name="Year 2009-2010")

import datetime as dt
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%Y-%m-%d %H:%M:%S")

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### NA SUM #####################")
    print(dataframe.isnull().sum().sum())
    print("##################### Describe #####################")
    print(dataframe.describe())
    print("##################### Nunique #####################")
    print(dataframe.nunique())

check_df(df)

df[df["Invoice"].str.contains("C",na=False)].head()

def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df

missing_values_analysis(df)

df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"]>0]
df.dropna(inplace=True)
df["TotalPrice"] = df["Quantity"] * df["Price"]

missing_values_analysis(df)

cltv_calculation = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                        'Quantity': lambda x: x.sum(),
                                        'TotalPrice': lambda x: x.sum()})

cltv_calculation.columns = ["total_transaction","total_unit","total_price"]
cltv_calculation.head()

##################################################
# 2. Average Order Value (average_order_value = total_price / total_transaction)
##################################################

cltv_calculation["avg_order_value"] = cltv_calculation["total_price"] / cltv_calculation["total_transaction"]

##################################################
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
##################################################

cltv_calculation["purchase_frequency"] = cltv_calculation["total_transaction"] / cltv_calculation.shape[0]

##################################################
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
##################################################

repeat_rate = cltv_calculation[cltv_calculation.total_transaction > 1].shape[0] / cltv_calculation.shape[0]
churn_rate = 1 - repeat_rate
print(f" The repeat rate for transaction ",repeat_rate)
print(f" The Churn rate is ",churn_rate)

##################################################
# 5. Profit Margin (profit_margin =  total_price * 0.10)
##################################################

cltv_calculation["profit_margin"] = cltv_calculation["total_price"] * 0.10

##################################################
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
##################################################

# Customer Value

cltv_calculation["customer_value"] = (cltv_calculation["avg_order_value"] * cltv_calculation["purchase_frequency"]) / churn_rate

##################################################
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
##################################################

cltv_calculation["cltv"] = cltv_calculation["customer_value"] * cltv_calculation["profit_margin"]

cltv_calculation.head()

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_calculation[["cltv"]])
cltv_calculation["scaled_cltv"] = scaler.transform(cltv_calculation[["cltv"]])
cltv_calculation.sort_values(by="scaled_cltv", ascending=False).head()

##################################################
# 8. Creating Segments based on CLTV Value of Customers
##################################################

cltv_calculation["segment"] = pd.qcut(cltv_calculation["scaled_cltv"], 4, labels=["D", "C", "B", "A"])
cltv_calculation.head()

cltv_calculation[["total_transaction", "total_unit", "total_price", "cltv", "scaled_cltv","segment"]].sort_values(by="scaled_cltv",
                                                                                              ascending=False).head()
# DIFFERENT APPROACH TO CALCULATE CLV

def get_month(x):
    return dt.datetime(x.year,x.month,1)

def get_date(df,column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year,month,day


def calculate_cohort(df):
    df["InvoiceMonth"] = df["InvoiceDate"].apply(get_month)

    grouping = df.groupby(["Customer ID"])["InvoiceMonth"]

    df["CohortMonth"] = grouping.transform("min")

    invoice_year, invoice_month, _ = get_date(df, "InvoiceMonth")

    cohort_year, cohort_month, _ = get_date(df, "CohortMonth")

    # Calculate difference in years

    years_diff = invoice_year - cohort_year

    # Calculate difference in months

    months_diff = invoice_month - cohort_month

    df["CohortIndex"] = years_diff * 12 + months_diff + 1
    cohort_data = df.groupby(["CohortMonth", "CohortIndex"])["Customer ID"].nunique()
    cohort_data = cohort_data.reset_index()
    cohort_counts = cohort_data.pivot(index="CohortMonth",
                                      columns="CohortIndex",
                                      values="Customer ID")

    cohort_sizes = cohort_counts.iloc[:, 0]
    retention = cohort_counts.divide(cohort_sizes, axis=0)
    retention_rate = retention.round(3) * 100
    return cohort_sizes, retention, retention_rate

cohort_sizes,retention,retention_rate = calculate_cohort(df)

retention

churn = 1 - retention
churn

# Calculating the mean of retention rate
retention_rate = retention.iloc[:,1:].mean().mean()

# Calculating the mean of churn rate

churn_rate = churn.iloc[:,1:].mean().mean()

# Print rounded retention and churn rates
print('Retention rate: {:.2f}; Churn rate: {:.2f}'.format(retention_rate, churn_rate))

df.head()

##################################################
# CALCULATING BASIC CLV

# Average Monthly Spent * Projected Customer Lifespan
##################################################

# Calculating monthly spending per customer
monthly_revenue = df.groupby(["Customer ID","InvoiceMonth"])["TotalPrice"].sum()

# Calculating average monthly spend
monthly_revenue = np.mean(monthly_revenue)
monthly_revenue

# Define lifespan to 36 months
lifespan_months = 36

# Calculate basic CLV
clv_basic = monthly_revenue * lifespan_months

# Print basic CLV value
print('Average basic CLV is {:.1f} USD'.format(clv_basic))

##################################################
# CALCULATING GRANULAR CLV

# It will focus on more granular data points at the invoice level.
##################################################

# Calculating average revenue per invoice
revenue_per_purchase = df.groupby(['Invoice'])['TotalPrice'].mean().mean()

# Calculating average number of unique invoices per customer per month
frequency_per_month = df.groupby(['Customer ID','InvoiceMonth'])['Invoice'].nunique().mean()

# Define lifespan to 36 months
lifespan_months = 36

# Calculating granular CLV
clv_granular = revenue_per_purchase * frequency_per_month * lifespan_months

# All together
print('Average granular CLV is {:.1f} USD'.format(clv_granular))
print('Revenue Per Purchase {:.1f} USD'.format(revenue_per_purchase))
print('Frequency Per Month {:.1f} USD'.format(frequency_per_month))

##################################################
# CALCULATING TRADITIONAL CLV
##################################################

# Calculate monthly spend per customer
monthly_revenue = df.groupby(['Customer ID','InvoiceMonth'])['TotalPrice'].sum().mean()

# Calculate average monthly retention rate
retention_rate = retention.iloc[:,1:].mean().mean()

# Calculate average monthly churn rate
churn_rate = 1 - retention_rate

# Calculate traditional CLV
clv_traditional = monthly_revenue * (retention_rate / churn_rate)

# Print traditional CLV and the retention rate values
print('Average traditional CLV is {:.1f} USD at {:.1f} % retention_rate'.format(clv_traditional, retention_rate*100))
