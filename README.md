
# Real-Time Traliling Stops Strategy Using Linear Regression 

**Objective:** Collected and trained Forex Currency data to build an optimized trailing-stops strategy to make real-time investment decisions.



## Installation

Install the necessary python packages 

```bash
  $ pip install -r requirements.txt
```
    
## Project Description 

1. Collected 40 hours of 8 currency pairs quotes data through polygon API. Cleaned, calculated, and updated key features in real-time.
    - Timestamp (𝑇)
    - Mean price (𝑃),
    - Maximum price (MAX),
    - Minimum price (MIN),
    - Volatility (VOL = (MAX–MIN)/𝑃),
    - Fractal dimension (FD) calculated with a counting process on a modified Ketner Channel 
    - Return (𝑅𝑖=(𝑃𝑖−𝑃𝑖−1)/𝑃i-1)

      ***Example of cleaned EUR-USD quotes (updated every 6 min):*** 
      <img src="./images/agg_table_image.png" width =600>
2. Trained and stored optimized regression models for each currency pair. (The models performance wasn't optimum due to small amount of training data.)
3. Built an optimized real-time trailing-stop-strategy, and used our model predictions to make real-time investment decisions. We used go long and go short strategys, and we will make investment decisions base on our model predictions, modeling errors, and actual returns.

      ***Example of EUR-USD models prediction table (updated every 6 min):***

      <img src="./images/ml_table_image.png" width =450>

      
      ***Example of EUR-USD trailing-stop investment strategy table (Go Long) (updated every hour):***
      <img src="./images/investment_result.png" width =450>

For the details of codes, Checked out [README.MD](https://github.com/Felix-only/-Real-Time-Trailing-Stops-Strategy-Using-Regression/blob/master/notebooks/README.md) in the **notebooks folder**.

The **output_csv folder** contains 10 hours of Forex currencies investments simulation results.

For the detail schema of the project, check out the PDF documents in the **detail_descriptions** folder.








 

