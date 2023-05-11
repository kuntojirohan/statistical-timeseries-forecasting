# statistical-timeseries-forecasting
The project examines various statistical time series forecasting methods using historical financial markets data. In particular, we have considered the S&P 500 Stock Index and the Bloomberg Barclays US Aggregate Bond Index.


## Running the Forecasting Analysis Script

- Open a terminal (Command Prompt on Windows or Terminal on macOS/Linux) and navigate to the directory containing the script.

- Install the required libraries and dependencies from `requirements.txt` by running the following command in your terminal:
```pip install -r requirements.txt ```

- To run the script with the rolling method and a window size of 240, use the following command: 
```python main.py --method rolling --window 240 ```

- Alternatively, to run the script with the recursive method, use the following command:
```python main.py --method recursive```