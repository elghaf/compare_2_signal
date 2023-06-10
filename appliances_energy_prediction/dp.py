import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read the data from the CSV file
path = "data\\energydata_complete.csv"
data = pd.read_csv(path)

# Split the data into features (X) and target variable (y)
X = data.drop(['date', 'Appliances'], axis=1)
y = data['Appliances']

# Perform feature selection using correlation analysis
correlation = data.corr()['Appliances'].abs().sort_values(ascending=False)
selected_features_corr = correlation[correlation > 0.1].index.tolist()
selected_features_corr.remove('Appliances')

# Perform feature selection using univariate feature selection
selector = SelectKBest(f_regression, k=10)
selector.fit(X, y)
selected_features_uni = X.columns[selector.get_support()].tolist()

# Combine selected features from both methods
selected_features = list(set(selected_features_corr + selected_features_uni))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)

# Instantiate and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Create a scatter plot of actual vs predicted values
plt.scatter(y_test, y_pred, color='b', alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Regression Model: Actual vs Predicted')
plt.show()
