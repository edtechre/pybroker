# Training A Model

Source: `docs/source/notebooks/6. Training a Model.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Training a Model

In the [last notebook](https://www.pybroker.com/en/latest/notebooks/5.%20Writing%20Indicators.html), we learned how to write stock indicators in **PyBroker**. Indicators are a good starting point for developing a trading strategy. But to create a successful strategy, it is likely that a more sophisticated approach using predictive modeling will be needed.

Fortunately, one of the main features of **PyBroker** is the ability to train and backtest machine learning models. These models can utilize indicators as features to make more accurate predictions about market movements. Once trained, these models can be backtested using a popular technique known as [Walkforward Analysis](https://www.youtube.com/watch?v=WBZ_Vv-iMv4), which simulates how a strategy would perform during actual trading.

We'll explain Walkforward Analysis more in depth later in this notebook. But first, let's get started with some needed imports!

```python
import numpy as np
import pandas as pd
import pybroker
from numba import njit
from pybroker import Strategy, StrategyConfig, YFinance
```

As with [DataSource](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.DataSource) and [Indicator](https://www.pybroker.com/en/latest/reference/pybroker.indicator.html#pybroker.indicator.Indicator) data, **PyBroker** can also cache trained models to disk. You can enable caching for all three by calling [pybroker.enable_caches](https://www.pybroker.com/en/latest/reference/pybroker.cache.html#pybroker.cache.enable_caches):

```python
pybroker.enable_caches('walkforward_strategy')
```

In [the last notebook](https://www.pybroker.com/en/latest/notebooks/5.%20Writing%20Indicators.html), we implemented an indicator that calculates the close-minus-moving-average (CMMA) using [NumPy](https://www.numpy.org) and [Numba](https://numba.pydata.org/). Here's the code for the CMMA indicator again:

```python
def cmma(bar_data, lookback):

    @njit  # Enable Numba JIT.
    def vec_cmma(values):
        # Initialize the result array.
        n = len(values)
        out = np.array([np.nan for _ in range(n)])
        
        # For all bars starting at lookback:
        for i in range(lookback, n):
            # Calculate the moving average for the lookback.
            ma = 0
            for j in range(i - lookback, i):
                ma += values[j]
            ma /= lookback
            # Subtract the moving average from value.
            out[i] = values[i] - ma
        return out
    
    # Calculate for close prices.
    return vec_cmma(bar_data.close)

cmma_20 = pybroker.indicator('cmma_20', cmma, lookback=20)
```

## Train and Backtest

Next, we want to build a model that predicts the next day's return using the 20-day CMMA. Using [simple linear regression](https://en.wikipedia.org/wiki/Simple_linear_regression) is a good approach to begin experimenting with. Below we import a [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) model from [scikit-learn](https://scikit-learn.org/stable/):

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
```

We create a ```train_slr``` function to train the ```LinearRegression``` model:

```python
def train_slr(symbol, train_data, test_data):
    # Train
    # Previous day close prices.
    train_prev_close = train_data['close'].shift(1)
    # Calculate daily returns.
    train_daily_returns = (train_data['close'] - train_prev_close) / train_prev_close
    # Predict next day's return.
    train_data['pred'] = train_daily_returns.shift(-1)
    train_data = train_data.dropna()
    # Train the LinearRegession model to predict the next day's return
    # given the 20-day CMMA.
    X_train = train_data[['cmma_20']]
    y_train = train_data[['pred']]
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Test
    test_prev_close = test_data['close'].shift(1)
    test_daily_returns = (test_data['close'] - test_prev_close) / test_prev_close
    test_data['pred'] = test_daily_returns.shift(-1)
    test_data = test_data.dropna()
    X_test = test_data[['cmma_20']]
    y_test = test_data[['pred']]
    # Make predictions from test data.
    y_pred = model.predict(X_test)
    # Print goodness of fit.
    r2 = r2_score(y_test, np.squeeze(y_pred))
    print(symbol, f'R^2={r2}')
    
    # Return the trained model and columns to use as input data.
    return model, ['cmma_20']
```

The ```train_slr``` function uses the 20-day CMMA as the input feature, or predictor, for the ```LinearRegression``` model. The function then fits the ```LinearRegression``` model to the training data for that stock symbol.

After fitting the model, the function uses the testing data to evaluate the model's accuracy, specifically by computing the [R-squared](https://en.wikipedia.org/wiki/Coefficient_of_determination) score. The R-squared score provides a measure of how well the ```LinearRegression``` model fits the testing data.

The final output of the ```train_slr``` function is the trained ```LinearRegression``` model specifically for that stock symbol, along with the ```cmma_20``` column, which is to be used as input data when making predictions. **PyBroker** will use this model to predict the next day's return of the stock during the backtest. The ```train_slr``` function will be called for each stock symbol, and the trained models will be used to predict the next day's return for each individual stock.

Once the function to train the model has been defined, it needs to be registered with **PyBroker**. This is done by creating a new [ModelSource](https://www.pybroker.com/en/latest/reference/pybroker.model.html#pybroker.model.ModelSource)  instance using the [pybroker.model](https://www.pybroker.com/en/latest/reference/pybroker.model.html#pybroker.model.model) function. The arguments to this function are the name of the model (```'slr'``` in this case), the function that will train the model (```train_slr```), and a list of indicators to use as inputs for the model (in this case, ```cmma_20```).

```python
model_slr = pybroker.model('slr', train_slr, indicators=[cmma_20])
```

To create a trading strategy that uses the trained model, a new [Strategy](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy) object is created using the [YFinance](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.YFinance) data source, and specifying the start and end dates for the backtest period.

```python
config = StrategyConfig(bootstrap_sample_size=100)
strategy = Strategy(YFinance(), '3/1/2017', '3/1/2022', config)
strategy.add_execution(None, ['NVDA', 'AMD'], models=model_slr)
```

The [add_execution](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.add_execution) method is then called on the [Strategy](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy) object to specify the details of the trading execution. In this case, a ```None``` value is passed as the first argument, which means that no trading function will be used during the backtest.

The last step is to run the backtest by calling the [backtest](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.backtest) method on the ```Strategy``` object, with a ```train_size``` of ```0.5``` to specify that the model should be trained on the first half of the backtest data, and tested on the second half.

```python
strategy.backtest(train_size=0.5)
```

## Walkforward Analysis


**PyBroker** employs a powerful algorithm known as [Walkforward Analysis](https://www.youtube.com/watch?v=WBZ_Vv-iMv4) to perform backtesting. The algorithm partitions the backtest data into a fixed number of time windows, each containing a train-test split of data.

The Walkforward Analysis algorithm then proceeds to "walk forward" in time, in the same manner that a trading strategy would be executed in the real world. The model is first trained on the earliest window and then evaluated on the test data in that window.

As the algorithm moves forward to evaluate the next window in time, the test data from the previous window is added to the training data. This process continues until all of the time windows are evaluated.

![Walkforward Diagram](https://github.com/edtechre/pybroker/blob/master/docs/_static/walkforward.png?raw=true)

By using this approach, the Walkforward Analysis algorithm is able to simulate the real-world performance of a trading strategy, and produce more reliable and accurate backtesting results.

Let's consider a trading strategy that generates buy and sell signals from the [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) model that we trained earlier. The strategy is implemented as the ```hold_long``` function:

```python
def hold_long(ctx):
    if not ctx.long_pos():
        # Buy if the next bar is predicted to have a positive return:
        if ctx.preds('slr')[-1] > 0:
            ctx.buy_shares = 100
    else:
        # Sell if the next bar is predicted to have a negative return:
        if ctx.preds('slr')[-1] < 0:
            ctx.sell_shares = 100
            
strategy.clear_executions()
strategy.add_execution(hold_long, ['NVDA', 'AMD'], models=model_slr)
```

The ```hold_long``` function opens a long position when the model predicts a positive return for the next bar, and then closes the position when the model predicts a negative return.

The [ctx.preds('slr')](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.preds) method is used to access the predictions made by the ```'slr'``` model for the current stock symbol being executed in the function (NVDA or AMD). The predictions are stored in a [NumPy array](https://numpy.org/doc/stable/reference/generated/numpy.array.html), and the most recent prediction for the current stock symbol is accessed using ```ctx.preds('slr')[-1]```, which is the model's prediction of the next bar's return.

Now that we have defined a trading strategy and registered the ```'slr'``` model, we can run the backtest using the Walkforward Analysis algorithm.

The backtest is run by calling the [walkforward](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.walkforward) method on the ```Strategy``` object, with the desired number of time windows and train/test split ratio. In this case, we will use 3 time windows, each with a 50/50 train-test split. 

Additionally, since our ```'slr'``` model makes a prediction for one bar in the future, we need to specify the ```lookahead``` parameter as ```1```. This is necessary to ensure that training data does not leak into the test boundary. The ```lookahead``` parameter should always be set to the number of bars in the future being predicted.

```python
result = strategy.walkforward(
    warmup=20, 
    windows=3, 
    train_size=0.5, 
    lookahead=1, 
    calc_bootstrap=True
)
```

During the backtesting process using the Walkforward Analysis algorithm, the ```'slr'``` model is trained on a given window's training data, and then the ```hold_long``` function runs on the same window's test data.

The model is trained on the training data to make predictions about the next day's returns. The ```hold_long``` function then uses these predictions to make buy or sell decisions for the current day's trading session.

By evaluating the performance of the trading strategy on the test data for each window, we can see how well the strategy is likely to perform in real-world trading conditions. This process is repeated for each time window in the backtest, using the results to evaluate the overall performance of the trading strategy:

```python
result.metrics_df
```

```python
result.bootstrap.conf_intervals
```

```python
result.bootstrap.drawdown_conf
```

In summary, we have now completed the process of training and backtesting a linear regression model using **PyBroker**, with the help of Walkforward Analysis. The metrics that we have seen are based on the test data from all of the time windows in the backtest. Although our trading strategy needs to be improved, we have gained a good understanding of how to train and evaluate a model in **PyBroker**.

Please keep in mind that before conducting regression analysis, it is important to verify certain assumptions such as [homoscedasticity](https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity), normality of residuals, etc. I have not provided the details for these assumptions here for the sake of brevity and recommend that you perform this exercise on your own.

We are also not limited to just building linear regression models in **PyBroker**. We can train other model types such as gradient boosted machines, neural networks, or any other architecture that we choose. This flexibility allows us to explore and experiment with various models and approaches to find the best performing model for our specific trading goals.

PyBroker also offers customization options, such as the ability to specify an [input_data_fn](https://www.pybroker.com/en/latest/reference/pybroker.model.html#pybroker.model.model) for our model in case we need to customize how its input data is built. This would be required when constructing input for autoregressive models (i.e. ARMA or RNN) that use multiple past values to make predictions. Similarly, we can specify our own [predict_fn](https://www.pybroker.com/en/latest/reference/pybroker.model.html#pybroker.model.model) to customize how predictions are made (by default, the model's ```predict``` function is called).

With this knowledge, you can start building and testing your own models and trading strategies in **PyBroker**, and begin exploring the vast possibilities that this framework offers!
