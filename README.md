# Smart Waste Management - Fill-Level Forecasting

An undergraduate machine learning project focused on forecasting smart waste-bin fill levels using time-series sensor data and deep learning models.

The project explores multiple neural-network architectures for predicting future bin fill levels, with the broader goal of supporting data-driven waste collection and management.

## Project Overview

Smart waste-management systems can use sensor data from waste bins to understand how quickly bins are filling and potentially improve collection planning.

In this project, time-series sensor data was processed and analyzed to model waste-bin fill-level behavior.

The workflow included:

- Processing raw smart-bin sensor data
- Exploring fill-level and battery measurements
- Preparing time-series data for forecasting
- Creating supervised learning sequences
- Training multiple neural-network architectures
- Generating fill-level predictions
- Comparing predicted values with actual observations
- Evaluating model performance using regression metrics

## Dataset

The raw smart-bin data contained sensor measurements including:

- Fill percentage
- Battery level
- Event date
- Update state
- Distance
- Bin identifier and type

The preprocessing workflow converted raw JSON sensor records into a structured CSV dataset for analysis and modeling.

Approximately **16,800 sensor records** were processed during this stage.

## Machine Learning Approach

Because bin fill level changes sequentially over time, the project explored models capable of learning temporal patterns from historical observations.

### LSTM

Long Short-Term Memory (LSTM) networks were used as the primary time-series forecasting approach.

Experiments included:

- Standard LSTM
- Different look-back windows
- Stateful LSTM / memory between batches
- Stacked LSTM architectures
- Different time-step configurations

### CNN

A one-dimensional Convolutional Neural Network (Conv1D) was explored for learning local patterns from sequential fill-level observations.

### CNN-LSTM

A hybrid CNN-LSTM architecture was also evaluated.

The CNN component extracts patterns from the input sequence, while the LSTM component models temporal dependencies before producing the fill-level prediction.

### MLP

A Multi-Layer Perceptron was included as an additional neural-network forecasting approach for comparison.

## Model Evaluation

The forecasting models were evaluated using regression metrics including:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R2 score

Predicted fill levels were also visualized against the actual time-series values to examine how well each architecture followed the underlying fill pattern.

## Example Results

The repository contains visualizations from several LSTM experiments, including:

- Baseline LSTM forecasting
- Alternative LSTM configurations
- Stateful LSTM
- Stacked LSTM
- Time-step experiments
- Window/look-back experiments

These plots provide a visual comparison between actual and predicted bin fill levels.

## Project Workflow

```text
Raw Sensor Data
      |
JSON / CSV Processing
      |
Data Exploration
      |
Time-Series Preparation
      |
Sequence / Window Creation
      |
Train-Test Split
      |
Model Training
      |
LSTM / CNN / CNN-LSTM / MLP
      |
Prediction
      |
Performance Evaluation
```

## Technologies

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Matplotlib
- Seaborn
- Plotly

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

The scripts inside the `src/` directory contain the preprocessing and model experiments.

For example:

```bash
python src/lstm_forecasting.py
```

Dataset files used by the scripts are stored in the `data/` directory.

## Academic Context

This project originated as an undergraduate B.Tech major project in Computer Science and Engineering - Data Science.

The work was completed as a team academic project exploring machine learning and deep learning techniques for smart waste-management applications.

This repository presents the machine-learning and time-series forecasting components of that work in a portfolio-friendly format.

## Key Learning Outcomes

Through this project, I gained hands-on experience with:

- Time-series data preprocessing
- Sensor-data analysis
- Feature and sequence preparation
- LSTM neural networks
- CNN-based sequence modeling
- Hybrid CNN-LSTM architectures
- Training and evaluating deep-learning models
- Regression evaluation metrics
- Comparing model predictions with actual time-series observations

## Future Improvements

Potential extensions include:

- Hyperparameter optimization
- Additional forecasting baselines
- Cross-validation designed for time-series data
- Improved model-performance comparison
- Real-time sensor-data ingestion
- Model deployment through an API
- Integration with route-optimization systems for waste collection
