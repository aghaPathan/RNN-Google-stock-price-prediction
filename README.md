# RNN Google Stock Price Prediction

A Recurrent Neural Network model for predicting Google stock prices using historical data.

## Overview

This project builds and fine-tunes an RNN (LSTM) model to predict stock prices based on time series data.

## Features

- 📈 **LSTM Architecture** — Long Short-Term Memory for sequence learning
- 🔧 **Hyperparameter Tuning** — Optimized for better accuracy
- 📊 **Visualization** — Predicted vs actual price comparison
- ✅ **Validation** — Train/test split for model evaluation

## Dataset

Historical Google stock prices including:
- Open, High, Low, Close prices
- Trading volume

## Model Architecture

```
Input Layer → LSTM Layers → Dropout → Dense → Output
```

## Requirements

- Python 3.x
- Keras/TensorFlow
- NumPy
- pandas
- matplotlib
- scikit-learn

## Usage

```bash
jupyter notebook RNN_Stock_Prediction.ipynb
```

## Key Steps

1. **Data Preprocessing** — Scaling, reshaping for LSTM input
2. **Model Building** — Stacked LSTM layers with dropout
3. **Training** — Fit on historical data
4. **Prediction** — Forecast future prices
5. **Evaluation** — Compare with actual prices

## Results

Model captures overall trends in stock price movement with tuned hyperparameters.

## License

MIT

---

## CI Status

All PRs are checked for:
- ✅ Syntax (Python, JS, TS, YAML, JSON, Dockerfile, Shell)
- ✅ Secrets (No hardcoded credentials)
- ✅ Security (High-severity vulnerabilities)

