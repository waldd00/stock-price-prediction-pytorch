# Stock Price Prediction with PyTorch (RNN / LSTM / GRU)

Daily stock closing price prediction with three recurrent models (RNN, LSTM, GRU).
The target is the next-day log-return, and price is reconstructed as
price = anchor * exp(return). This avoids the extrapolation problem you get when
predicting absolute price on a trending series. Includes a Streamlit demo app.

## Features
- Stationary inputs: log-return, Close/SMA_10, Close/SMA_50, RSI, log-volume
- Chronological train / val / test split, scalers fit on train only
- Early stopping on validation loss
- Metrics: RMSE, MAE, MAPE, directional accuracy, plus a naive baseline
- Recursive multi-step forecast
- Streamlit app for any ticker

## Structure
```
src/        data.py, model.py, engine.py, forecast.py
scripts/    train.py
app/        streamlit_app.py
notebooks/  stock_prediction_pytorch.ipynb
outputs/    models, metrics, figures (generated)
```

## Setup
```
pip install -r requirements.txt
pip install jupyter          # only if you want to run the notebook
```

## Usage
```
python scripts/train.py
streamlit run app/streamlit_app.py
jupyter notebook notebooks/stock_prediction_pytorch.ipynb
```

## Testing
```
pip install -r requirements-dev.txt
pytest
```
Tests use synthetic price data (no network calls) and cover feature engineering,
dataset splitting/scaling, model output shapes, metrics edge cases, and training
reproducibility.

## Results
| Model | RMSE | MAE  | MAPE % | DirAcc % |
|-------|------|------|--------|----------|
| RNN   | 3.40 | 2.48 | 1.81   | 48.14    |
| LSTM  | 3.65 | 2.66 | 1.98   | 49.17    |
| GRU   | 3.39 | 2.46 | 1.79   | 48.97    |
| Naive | 3.36 | 2.44 | 1.77   | -        |

The models land close to the naive baseline on RMSE, which is expected for daily price
data. Directional accuracy near 50% means no real signal beyond chance; above 53% means
a weak signal. Not investment advice.

## License
MIT
