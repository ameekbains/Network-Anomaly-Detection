# Network-Anomaly-Detection
This project implements an anomaly detection system for monitoring network performance. Using simulated network data, the system employs the Isolation Forest machine learning algorithm to detect anomalies (outliers) such as sudden spikes in latency, signal strength fluctuations, packet loss, and interference. The system also supports real-time anomaly detection and provides model evaluation metrics.
## Features:
- Simulates network performance data over time.
- Detects anomalies using the Isolation Forest algorithm.
- Provides real-time anomaly detection.
- Evaluates the model using classification metrics.

## Requirements

- Python 3.x
- Libraries: Pandas, NumPy, scikit-learn, Matplotlib

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/ameekbains/Network-Anomaly-Detection.git
    cd Network-Anomaly-Detection

    ```

2. Create a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Running the Anomaly Detection Script

The main script for training the anomaly detection model is located in the `src/anomaly_detection.py` file. You can run it directly using:

```bash
python src/anomaly_detection.py
