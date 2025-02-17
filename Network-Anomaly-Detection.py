import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time

# Simulating network performance data (e.g., Signal Strength, Latency, Error Rate, Packet Loss, Interference Level)
# Simulating data for 1000 minutes
data = {
    'timestamp': pd.date_range(start='2025-01-01', periods=1000, freq='T'),
    'Signal Strength (dBm)': np.random.normal(0, 1, 1000),  # Normally distributed data
    'Latency (ms)': np.random.normal(20, 5, 1000),  # Normally distributed data
    'Error Rate': np.random.normal(0.01, 0.005, 1000),  # Normally distributed data
    'Packet Loss (%)': np.random.normal(0.5, 0.2, 1000),  # Normally distributed data
    'Interference Level': np.random.normal(50, 10, 1000)  # Normally distributed data
}

# Create a DataFrame
df = pd.DataFrame(data)

# Simulate anomalies (outliers)
anomalies = pd.DataFrame({
    'timestamp': pd.date_range(start='2025-01-01', periods=10, freq='D'),
    'Signal Strength (dBm)': np.random.uniform(-100, -50, 10),
    'Latency (ms)': np.random.uniform(100, 300, 10),
    'Error Rate': np.random.uniform(0.05, 0.1, 10),
    'Packet Loss (%)': np.random.uniform(10, 30, 10),
    'Interference Level': np.random.uniform(80, 100, 10)
})

# Append anomalies to the data
df = df.append(anomalies, ignore_index=True)

# 1. Data Preprocessing: Handle missing values and normalize the features
df.fillna(method='ffill', inplace=True)

# Normalize the features
scaler = StandardScaler()
features = ['Signal Strength (dBm)', 'Latency (ms)', 'Error Rate', 'Packet Loss (%)', 'Interference Level']
df[features] = scaler.fit_transform(df[features])

# Drop the timestamp as it's not used for modeling
df.drop('timestamp', axis=1, inplace=True)

# 2. Train the Anomaly Detection Model (Isolation Forest)
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)  # Contamination is the expected percentage of anomalies
model.fit(df[features])

# Predict anomalies
df['anomaly'] = model.predict(df[features])  # -1 indicates anomaly, 1 indicates normal

# Convert anomalies from -1/1 to True/False for easier understanding
df['anomaly'] = df['anomaly'].map({-1: True, 1: False})

# Visualizing the anomalies (Signal Strength with anomalies highlighted)
plt.figure(figsize=(12, 6))
plt.plot(df['Signal Strength (dBm)'], label='Signal Strength (dBm)')
plt.scatter(df.index[df['anomaly'] == True], df['Signal Strength (dBm)'][df['anomaly'] == True], color='red', label='Anomalies')
plt.legend()
plt.title('Network Signal Strength with Anomalies')
plt.xlabel('Time')
plt.ylabel('Signal Strength (dBm)')
plt.show()

# 3. Evaluate the Model (Example with true anomalies)
# For this example, we assume the 'anomalies' dataframe contains true anomalies
true_anomalies = [True] * 10  # Simulated true anomalies
pred_anomalies = df['anomaly'].tail(10).tolist()  # Predicted anomalies from the last 10 rows (simulated anomalies)

# Print classification report for evaluation
print("Classification Report for Anomaly Detection:\n")
print(classification_report(true_anomalies, pred_anomalies))

# 4. Real-Time Monitoring (simulated by checking the last few points in the data)
# Simulate incoming data for real-time anomaly detection
for i in range(1000, len(df)):
    new_data = df.iloc[i:i+1][features]
    
    # Predict anomaly in real-time
    anomaly = model.predict(new_data)
    
    # Trigger alert if anomaly is detected
    if anomaly == -1:
        print(f"ALERT: Anomaly detected at timestamp {df['timestamp'].iloc[i]}")
    
    # Simulate waiting for the next data point (1 second delay for simulation)
    time.sleep(1)

