import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Historical Zika virus cases data
data = {
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Cases': [100, 150, 2000, 1200, 700, 500, 300, 250, 200, 180],
    'Temp': [28.0, 28.5, 29.0, 28.8, 28.6, 28.4, 28.2, 28.0, 27.8, 27.6],
    'Precip': [120, 130, 150, 140, 135, 130, 125, 120, 115, 110],
    'Mosquito_Pop': [50, 55, 70, 60, 55, 50, 45, 40, 35, 30],
    'Intervention': [0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}

# Convert data to numpy arrays
years = np.array(data['Year']).reshape(-1, 1)
cases = np.array(data['Cases'])
temp = np.array(data['Temp']).reshape(-1, 1)
precip = np.array(data['Precip']).reshape(-1, 1)
mosquito_pop = np.array(data['Mosquito_Pop']).reshape(-1, 1)
intervention = np.array(data['Intervention']).reshape(-1, 1)

# Create the feature matrix and target variable
X = np.hstack((years, temp, precip, mosquito_pop, intervention))
y = cases

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
multi_var_model = LinearRegression()
multi_var_model.fit(X_train, y_train)

# Predict future cases for the next 5 years with hypothetical future values
future_data = {
    'Year': [2024, 2025, 2026, 2027, 2028],
    'Temp': [27.5, 27.4, 27.3, 27.2, 27.1],
    'Precip': [105, 100, 95, 90, 85],
    'Mosquito_Pop': [25, 20, 15, 10, 5],
    'Intervention': [1.0, 1.1, 1.2, 1.3, 1.4]
}

future_years = np.array(future_data['Year']).reshape(-1, 1)
future_temp = np.array(future_data['Temp']).reshape(-1, 1)
future_precip = np.array(future_data['Precip']).reshape(-1, 1)
future_mosquito_pop = np.array(future_data['Mosquito_Pop']).reshape(-1, 1)
future_intervention = np.array(future_data['Intervention']).reshape(-1, 1)

future_X = np.hstack((future_years, future_temp, future_precip, future_mosquito_pop, future_intervention))
predicted_future_cases = multi_var_model.predict(future_X)

# Plotting
plt.figure(figsize=(12, 8))

# Cases
plt.subplot(2, 1, 1)
plt.scatter(data['Year'], data['Cases'], color='blue', label='Historical Cases')
plt.plot(data['Year'], data['Cases'], color='blue')
plt.scatter(future_data['Year'], predicted_future_cases, color='red', label='Predicted Cases')
plt.plot(future_data['Year'], predicted_future_cases, color='red', linestyle='dashed')
plt.xlabel('Year')
plt.ylabel('Number of Cases')
plt.title('Zika Virus Cases Prediction')
plt.legend()
plt.grid(True)

# Temperature, Precipitation, Mosquito Population, and Intervention Effectiveness
plt.subplot(2, 1, 2)
plt.plot(data['Year'], data['Temp'], color='orange', label='Temp (°C)')
plt.plot(future_data['Year'], future_data['Temp'], color='orange', linestyle='dashed')
plt.plot(data['Year'], data['Precip'], color='green', label='Precip (mm)')
plt.plot(future_data['Year'], future_data['Precip'], color='green', linestyle='dashed')
plt.plot(data['Year'], data['Mosquito_Pop'], color='purple', label='Mosquito Pop (index)')
plt.plot(future_data['Year'], future_data['Mosquito_Pop'], color='purple', linestyle='dashed')
plt.plot(data['Year'], data['Intervention'], color='brown', label='Intervention Effectiveness (index)')
plt.plot(future_data['Year'], future_data['Intervention'], color='brown', linestyle='dashed')
plt.xlabel('Year')
plt.ylabel('Index / °C / mm')
plt.title('Factors Influencing Zika Virus Spread')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

