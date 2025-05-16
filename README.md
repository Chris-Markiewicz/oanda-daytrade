## Implemented Strategies

### Machine Learning Based

-   **Current Model:** The primary strategy uses a model (e.g., LightGBM, RandomForest, Logistic Regression as selected by `ModelTrainer.py`) trained on lagged returns and potentially other features.
-   **Features Used:** Currently, the model is trained using 5 lagged 1-minute log returns as features.
-   **Prediction Target:** Predicts the direction (up/down: +1/-1) of the price for the next 1-minute bar.

### (Example) Contrarian Strategy

-   `ConTrader.py` implements a simple contrarian strategy based on the rolling mean of returns. This serves as a basic example of a live trading bot structure using the `tpqoa` library.

## Backtesting Framework

-   **Vectorized Backtesting:**
    -   `VectorizedMetrics.py`: Contains functions for fast, array-based calculations of strategy returns (including cost modeling) and Sharpe Ratio.
    -   `VectorizedPerformanceCalculator.py`: A class that utilizes `VectorizedMetrics.py` to evaluate the financial performance (P&L, Sharpe Ratio, equity curve) of ML model predictions. This is the primary backtesting engine used in `ModelTrainer.py` for validation and test set evaluations due to its speed.
-   **Iterative Backtesting (Legacy/Alternative):**
    -   `IterativeBase.py` and `IterativeBacktest.py`: Provide a bar-by-bar iterative backtesting framework. While more flexible for certain path-dependent strategies, it is significantly slower than the vectorized approach and is not the primary method used for ML model evaluation in the current `ModelTrainer.py` script.

## Key Technologies

-   **Python:** Core programming language.
-   **OANDA v20 API:** Used for fetching market data and potentially for trade execution (though live execution is primarily in `MLTrader.py` and `ConTrader.py`).
-   **`tpqoa`:** Python wrapper for the OANDA v20 API (the version used in this project is a local `tpqoa.py` module).
-   **Pandas:** For data manipulation, time series analysis, and DataFrame management.
-   **NumPy:** For numerical operations and array manipulations.
-   **Scikit-learn:** For machine learning models (Logistic Regression, Random Forest), evaluation metrics (F1-score, accuracy, classification report, confusion matrix), and utilities like `GridSearchCV` and `TimeSeriesSplit`.
-   **LightGBM:** A gradient boosting framework used for one of the machine learning models, known for its efficiency and performance.
-   **Matplotlib & Seaborn:** For generating plots, specifically confusion matrices and equity curves.
-   **Pickle:** For serializing and deserializing (saving and loading) trained machine learning models.

## Future Enhancements

-   Implement more sophisticated feature engineering (e.g., technical indicators like RSI, MACD; volatility measures like ATR; time-based features).
-   Explore different machine learning model architectures (e.g., LSTMs, GRUs for deeper time series analysis, or advanced ensembling techniques).
-   Integrate robust risk management into the live trading logic (`MLTrader.py`), such as dynamic stop-loss, take-profit orders, and more sophisticated position sizing rules.
-   Further refine the `VectorizedPerformanceCalculator.py` to include more detailed financial metrics (e.g., Sortino Ratio, Max Drawdown, Calmar Ratio).
-   Develop a user interface or dashboard for monitoring live trading performance and strategy parameters.
-   Implement comprehensive logging for both training and live trading activities for better traceability and debugging.
-   Containerize the application using Docker for easier setup, deployment, and scalability.
-   Investigate methods for more direct optimization of financial metrics (like Sharpe Ratio) within `GridSearchCV`, potentially by creating a highly optimized, simplified vectorized backtester suitable for use as a custom scorer.

## Disclaimer

**Trading financial instruments, including Forex and CFDs, involves substantial risk of loss and is not suitable for all investors. The software and strategies provided in this repository are for educational and illustrative purposes only. They do not constitute financial advice.**

**Past performance is not indicative of future results. Use this software at your own risk. The authors and contributors are not liable for any financial losses incurred.**

**Always test thoroughly in a practice/demo account before considering live trading with real funds.**
