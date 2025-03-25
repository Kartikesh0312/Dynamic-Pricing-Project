import pandas as pd
import matplotlib.pyplot as plt

# Load the sales data
sales_data = pd.read_csv('train.csv')
print(sales_data.head())  # Display the first 5 rows

# Load the features data (contains temperature, holidays, etc.)
features_data = pd.read_csv('features.csv')
print(features_data.head())

# Merge the sales and features datasets
merged_data = pd.merge(sales_data, features_data, on=['Store', 'Date'], how='left')
print(merged_data.head())

# Drop IsHoliday_y and rename IsHoliday_x to IsHoliday
merged_data = merged_data.drop(columns=['IsHoliday_y'])
merged_data = merged_data.rename(columns={'IsHoliday_x': 'IsHoliday'})

# Check for missing values
print(merged_data.isnull().sum())

# Fill missing values in MarkDown columns with 0
for col in ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
    merged_data[col] = merged_data[col].fillna(0)

# Fill missing values in other numerical columns with the mean
merged_data['Fuel_Price'] = merged_data['Fuel_Price'].fillna(merged_data['Fuel_Price'].mean())
merged_data['CPI'] = merged_data['CPI'].fillna(merged_data['CPI'].mean())
merged_data['Unemployment'] = merged_data['Unemployment'].fillna(merged_data['Unemployment'].mean())

# Check again for missing values
print(merged_data.isnull().sum())

# Convert Date column to datetime
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

# Create IsHolidaySeason feature (True for November and December)
merged_data['IsHolidaySeason'] = merged_data['Date'].dt.month.isin([11, 12])

# Display the first 5 rows to see the new feature
print(merged_data[['Date', 'IsHolidaySeason']].head())

# Create Demand feature based on Weekly_Sales
# Handle negative Weekly_Sales (set to 0)
merged_data['Weekly_Sales'] = merged_data['Weekly_Sales'].clip(lower=0)

# Define thresholds for Low, Medium, and High demand
low_threshold = merged_data['Weekly_Sales'].quantile(0.25)  # 25th percentile
high_threshold = merged_data['Weekly_Sales'].quantile(0.75)  # 75th percentile
print(f"Low Threshold (25th percentile): {low_threshold}")
print(f"High Threshold (75th percentile): {high_threshold}")
print(merged_data['Weekly_Sales'].describe())

# Categorize Weekly_Sales into Low, Medium, High
def categorize_demand(sales):
    if sales <= low_threshold:
        return 'Low'
    elif sales <= high_threshold:
        return 'Medium'
    else:
        return 'High'

merged_data['Demand'] = merged_data['Weekly_Sales'].apply(categorize_demand)

# Display the first 5 rows to see the new feature
print(merged_data[['Weekly_Sales', 'Demand']].head())
print(merged_data['Demand'].value_counts())

# Print column names to debug
print("Columns in merged_data:", merged_data.columns.tolist())

# Calculate summary statistics for Weekly_Sales by Demand, IsHoliday, and IsHolidaySeason
print("\nAverage Weekly Sales by Demand:")
print(merged_data.groupby('Demand')['Weekly_Sales'].mean())

print("\nAverage Weekly Sales by IsHoliday:")
print(merged_data.groupby('IsHoliday')['Weekly_Sales'].mean())

print("\nAverage Weekly Sales by IsHolidaySeason:")
print(merged_data.groupby('IsHolidaySeason')['Weekly_Sales'].mean())

# Creating bar plots to visualize the relationships.

# Plot 1: Average Weekly Sales by Demand
plt.figure(figsize=(8, 6))
avg_sales_demand = merged_data.groupby('Demand')['Weekly_Sales'].mean()
avg_sales_demand.plot(kind='bar', color='skyblue')
plt.title('Average Weekly Sales by Demand')
plt.xlabel('Demand')
plt.ylabel('Average Weekly Sales')
plt.xticks(rotation=0)
plt.show()

# Plot 2: Average Weekly Sales by IsHoliday
plt.figure(figsize=(8, 6))
avg_sales_holiday = merged_data.groupby('IsHoliday')['Weekly_Sales'].mean()
avg_sales_holiday.plot(kind='bar', color='lightgreen')
plt.title('Average Weekly Sales by IsHoliday')
plt.xlabel('IsHoliday')
plt.ylabel('Average Weekly Sales')
plt.xticks(rotation=0)
plt.show()

# Plot 3: Average Weekly Sales by IsHolidaySeason
plt.figure(figsize=(8, 6))
avg_sales_holiday_season = merged_data.groupby('IsHolidaySeason')['Weekly_Sales'].mean()
avg_sales_holiday_season.plot(kind='bar', color='salmon')
plt.title('Average Weekly Sales by IsHolidaySeason')
plt.xlabel('IsHolidaySeason')
plt.ylabel('Average Weekly Sales')
plt.xticks(rotation=0)
plt.show()

# Explore the relationship between Temperature and Weekly_Sales

# Scatter plot of Temperature vs. Weekly_Sales
plt.figure(figsize=(8, 6))
plt.scatter(merged_data['Temperature'], merged_data['Weekly_Sales'], alpha=0.5, color='purple')
plt.title('Temperature vs. Weekly Sales')
plt.xlabel('Temperature (Fahrenheit)')
plt.ylabel('Weekly Sales')
plt.show()

# Bin Temperature into categories (Cold, Mild, Hot)
bins = [0, 40, 70, 100]  # Temperature bins: Cold (<=40), Mild (40-70), Hot (>70)
labels = ['Cold', 'Mild', 'Hot']
merged_data['Temperature_Category'] = pd.cut(merged_data['Temperature'], bins=bins, labels=labels, include_lowest=True)

# Calculate average Weekly_Sales by Temperature_Category
print("\nAverage Weekly Sales by Temperature Category:")
avg_sales_temp = merged_data.groupby('Temperature_Category')['Weekly_Sales'].mean()
print(avg_sales_temp)

# Bar plot of Average Weekly Sales by Temperature Category
plt.figure(figsize=(8, 6))
avg_sales_temp.plot(kind='bar', color='orange')
plt.title('Average Weekly Sales by Temperature Category')
plt.xlabel('Temperature Category')
plt.ylabel('Average Weekly Sales')
plt.xticks(rotation=0)
plt.show()

# Step 6: Prepare Data for Modeling

# Encode categorical features (convert categories to numbers)
# Convert Demand (Low, Medium, High) to numerical values: Low=0, Medium=1, High=2
merged_data['Demand_Encoded'] = merged_data['Demand'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Convert IsHoliday and IsHolidaySeason (True/False) to 1/0
merged_data['IsHoliday_Encoded'] = merged_data['IsHoliday'].astype(int)
merged_data['IsHolidaySeason_Encoded'] = merged_data['IsHolidaySeason'].astype(int)

# Select features for the model
# We'll use Demand_Encoded, IsHoliday_Encoded, IsHolidaySeason_Encoded, and Temperature to predict Weekly_Sales
features = ['Demand_Encoded', 'IsHoliday_Encoded', 'IsHolidaySeason_Encoded', 'Temperature']
X = merged_data[features]  # Features (independent variables)
y = merged_data['Weekly_Sales']  # Target (dependent variable)

# Split the data into training and testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShape of training data:", X_train.shape)
print("Shape of testing data:", X_test.shape)

# Step 7: Build a Linear Regression Model

# Import the linear regression model from sklearn
from sklearn.linear_model import LinearRegression

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the model coefficients
print("\nModel Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef}")
print(f"Intercept: {model.intercept_}")

# 7.2: Random Forest Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Initialize the Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100],  # Number of trees
    'max_depth': [10, 20],      # Maximum depth of trees
}

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best Random Forest model
best_rf_model = grid_search.best_estimator_
print("\nBest Random Forest Parameters:", grid_search.best_params_)

# Make predictions with the best Random Forest model
rf_y_pred = best_rf_model.predict(X_test)

# Print feature importances for Random Forest
print("\nRandom Forest Feature Importances:")
for feature, importance in zip(features, best_rf_model.feature_importances_):
    print(f"{feature}: {importance}")

# Step 8: Evaluate the Model

# Import evaluation metrics from sklearn
from sklearn.metrics import mean_absolute_error, r2_score

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Diagonal line
plt.title('Actual vs. Predicted Weekly Sales')
plt.xlabel('Actual Weekly Sales')
plt.ylabel('Predicted Weekly Sales')
plt.show()

# Step 9: Summary of Findings
print("\nSummary of Findings:")
print("1. Demand has a strong impact on Weekly Sales (High demand leads to much higher sales).")
print("2. Sales are slightly higher on holidays and during the holiday season (November-December).")
print("3. Temperature has a moderate impact on sales—sales are highest in cold temperatures (≤40°F) and lowest in hot temperatures (>70°F).")
print("4. The linear regression model provides a baseline for predicting Weekly Sales.")
print("Next Steps: Try more advanced models (e.g., Random Forest), add more features, and tune the model for better performance.")