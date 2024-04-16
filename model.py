import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load datasets
sales_df = pd.read_csv("sales.csv")
stock_df = pd.read_csv("sensor_stock_levels.csv")
temp_df = pd.read_csv("sensor_storage_temperature.csv")

# Drop unnecessary columns
sales_df.drop(columns=["Unnamed: 0"], inplace=True)
stock_df.drop(columns=["Unnamed: 0"], inplace=True)
temp_df.drop(columns=["Unnamed: 0"], inplace=True)


def convert_to_datetime(data: pd.DataFrame = None, column: str = None):
    """Convert timestamp column to datetime format."""
    dummy = data.copy()
    dummy[column] = pd.to_datetime(dummy[column], format='%Y-%m-%d %H:%M:%S')
    return dummy


# Convert timestamp columns to datetime
sales_df = convert_to_datetime(sales_df, 'timestamp')
stock_df = convert_to_datetime(stock_df, 'timestamp')
temp_df = convert_to_datetime(temp_df, 'timestamp')


def convert_timestamp_to_hourly(data: pd.DataFrame = None, column: str = None):
    """Convert timestamp to hourly format."""
    dummy = data.copy()
    dummy[column] = pd.to_datetime(dummy[column].dt.floor('H'))
    return dummy


# Convert timestamp to hourly format
sales_df = convert_timestamp_to_hourly(sales_df, 'timestamp')
stock_df = convert_timestamp_to_hourly(stock_df, 'timestamp')
temp_df = convert_timestamp_to_hourly(temp_df, 'timestamp')

# Aggregate data
sales_agg = sales_df.groupby(['timestamp', 'product_id']).agg({'quantity': 'sum'}).reset_index()
stock_agg = stock_df.groupby(['timestamp', 'product_id']).agg({'estimated_stock_pct': 'mean'}).reset_index()
temp_agg = temp_df.groupby(['timestamp']).agg({'temperature': 'mean'}).reset_index()

# Merge dataframes
merged_df = stock_agg.merge(sales_agg, on=['timestamp', 'product_id'], how='left')
merged_df = merged_df.merge(temp_agg, on='timestamp', how='left')
merged_df['quantity'] = merged_df['quantity'].fillna(0)

# Merge additional features
product_categories = sales_df[['product_id', 'category']].drop_duplicates()
product_price = sales_df[['product_id', 'unit_price']].drop_duplicates()
merged_df = merged_df.merge(product_categories, on="product_id", how="left")
merged_df = merged_df.merge(product_price, on="product_id", how="left")

# Extract temporal features
merged_df['timestamp_day_of_month'] = merged_df['timestamp'].dt.day
merged_df['timestamp_day_of_week'] = merged_df['timestamp'].dt.dayofweek
merged_df['timestamp_hour'] = merged_df['timestamp'].dt.hour
merged_df.drop(columns=['timestamp'], inplace=True)

# One-hot encode categorical variables
merged_df = pd.get_dummies(merged_df, columns=['category'])
merged_df.drop(columns=['product_id'], inplace=True)

# Define X and y
X = merged_df.drop(columns=['estimated_stock_pct'])
y = merged_df['estimated_stock_pct']

# Define hyperparameters
max_depth = 8
min_samples_leaf = 4
min_samples_split = 8
n_estimators = 200
K = 10
split = 0.75
accuracy = []

# Model training and evaluation
for fold in range(0, K):
    # Instantiate RandomForestRegressor with specified hyperparameters
    model = RandomForestRegressor(max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf,
                                   min_samples_split=min_samples_split,
                                   n_estimators=n_estimators)
    scaler = StandardScaler()

    # Create training and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

    # Scale data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    trained_model = model.fit(X_train, y_train)

    # Generate predictions on test sample
    y_pred = trained_model.predict(X_test)

    # Compute mean absolute error
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    accuracy.append(mae)
    print(f"Fold {fold + 1}: MAE = {mae:.3f}")

print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")


