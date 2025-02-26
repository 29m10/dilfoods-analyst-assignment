import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load the dataset
sales_data = pd.read_csv("./sales_data_food.csv")
cleaned_sales_data = pd.read_csv("./cleaned_sales_data.csv")

def data_clean():

    # Display basic info
    print(sales_data.info())

    # Check for missing values
    print(sales_data.isnull().sum())

    # Fill missing values (if needed)
    sales_data.fillna(0, inplace=True)

    # Ensure correct data types
    sales_data["order_date"] = pd.to_datetime(sales_data["order_date"])
    sales_data["quantity"] = sales_data["quantity"].astype(int)
    sales_data["price"] = sales_data["price"].astype(float)

    # Save cleaned data
    sales_data.to_csv("./cleaned_sales_data.csv", index=False)

    return True

def calculate_revenue():

    # Add total_price column
    cleaned_sales_data["total_price"] = cleaned_sales_data["quantity"] * cleaned_sales_data["price"]

    # Total revenue per product category
    category_revenue = cleaned_sales_data.groupby("category")["total_price"].sum().reset_index()

    print(category_revenue)

    return True

def customer_retention():

    # Count number of orders per customer
    customer_orders = cleaned_sales_data.groupby("customer_id")["order_id"].nunique()

    # Identify repeat customers (more than 2 orders)
    repeat_customers = customer_orders[customer_orders > 2]
    print(repeat_customers)

    return True

def sales_trend():

    print(cleaned_sales_data.tail(10))

    # Convert 'order_date' to datetime format
    cleaned_sales_data["order_date"] = pd.to_datetime(cleaned_sales_data["order_date"])

    # Aggregate monthly revenue
    cleaned_sales_data["month"] = cleaned_sales_data["order_date"].dt.to_period("M")
    monthly_sales = cleaned_sales_data.groupby("month")["price"].sum()

    # Plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=monthly_sales.index.astype(str), y=monthly_sales.values)
    plt.xticks(rotation=45)
    plt.xlabel("Month")
    plt.ylabel("Total Revenue")
    plt.title("Monthly Sales Trend")
    plt.savefig("monthly_sales_trend.png", dpi=300, bbox_inches="tight")
    plt.show()

    return True


def top_5_customers_by_revenue():

    # Identify top 5 customers
    top_customers = cleaned_sales_data.groupby("customer_id")["price"].sum().nlargest(5)
    print(top_customers)

    return True

def category_performance():

    category_revenue = cleaned_sales_data.groupby("category")["price"].sum().reset_index()

    # Plot revenue by category
    plt.figure(figsize=(8, 5))
    sns.barplot(x=category_revenue["category"], y=category_revenue["price"])
    plt.xlabel("Product Category")
    plt.ylabel("Total Revenue")
    plt.title("Category Performance")
    plt.xticks(rotation=45)
    plt.savefig("category_performance.png", dpi=300, bbox_inches="tight")
    plt.show()

    return True

def predict_future_sales():

    # Convert 'order_date' to datetime format
    cleaned_sales_data["order_date"] = pd.to_datetime(cleaned_sales_data["order_date"])

    # Feature Engineering: Extract Year and Month
    cleaned_sales_data["year"] = cleaned_sales_data["order_date"].dt.year
    cleaned_sales_data["month"] = cleaned_sales_data["order_date"].dt.month

    # Encode categorical variables (Product Category)
    label_encoder = LabelEncoder()
    cleaned_sales_data["category_encoded"] = label_encoder.fit_transform(cleaned_sales_data["category"])

    # Prepare input features (year, month, category)
    X = cleaned_sales_data[["year", "month", "category_encoded"]]
    y = cleaned_sales_data["quantity"] * cleaned_sales_data["price"]  # Total revenue

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Generate next 12 months
    latest_date = pd.to_datetime(sales_data["order_date"].max())  # Convert to datetime
    future_dates = [(latest_date + timedelta(days=30 * i)).replace(day=1) for i in range(1, 13)]

    # Create future data frame
    future_df = pd.DataFrame({
        "year": [date.year for date in future_dates],
        "month": [date.month for date in future_dates]
    })

    # Predict revenue for each category separately
    category_names = label_encoder.classes_
    forecast_data = []

    for category in category_names:
        category_code = label_encoder.transform([category])[0]  # Get encoded category value
        future_df["category_encoded"] = category_code
        predicted_revenue = model.predict(future_df)

        # Store results
        for i, date in enumerate(future_dates):
            forecast_data.append([date.year, date.month, category, predicted_revenue[i]])

    # Convert to DataFrame
    forecast_df = pd.DataFrame(forecast_data, columns=["year", "month", "category", "predicted_revenue"])

    # Save to CSV
    output_file_path = "forecasted_revenue.csv"
    forecast_df.to_csv(output_file_path, index=False)

    return True