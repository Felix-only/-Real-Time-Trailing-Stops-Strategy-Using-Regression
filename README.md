
# Real-Time Traliling Stops Strategy Using Linear Regression 

**Objective:** Collected and trained Forex Currency data to build an optimize trailing-stops strategy to make real-time investment decisions.



## Installation

Install the necessary python packages 

```bash
  $ pip install -r requirements.txt
```
    
## Project Description 

1. Collected 40 hours of 8 currency pairs data through polygon API, calculated and updated key features in real-time.
    - Timestamp (𝑇)
    - Mean price (𝑃),
    - Maximum price (MAX),
    - Minimum price (MIN),
    - Volatility (VOL = (MAX–MIN)/𝑃),
    - Fractal dimension (FD) calculated with a counting process on a modified Ketner Channel 
    - Return (𝑅𝑖=(𝑃𝑖−𝑃𝑖−1)/𝑃i-1)
2. Trained and stored optimized regression models for each currency pair. (The models performance wasn't optimum due to small amount of training data.)
3. Built an optimized real-time trailing-stop-strategy, and used our model prediction to make real-time investment decisions. We used go long and go short strategys, and we will make investment decision base on our model predictions, our modeling errors, and our actual returns.

For details of codes, Checked out [README.MD](https://github.com/Felix-only/-Real-Time-Trailing-Stops-Strategy-Using-Regression/blob/master/notebooks/README.md) in the **notebooks folder**.

The **output_csv folder** contains the 10 hours Forex currencies investments simulation results.









 

