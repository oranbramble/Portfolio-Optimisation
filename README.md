# Portfolio-Optimisation

## Description

This script takes any number of 'assets' and finds the optimal weighting of investment on each stock to provide maximum returns. Each 'asset' is a stock of a company that is accessible on the stock market. For convenience, I have already included 5 assets so that this code can be run immediately. These 5 assets are:

- Auto Trader
- Experian
- Rightmove
- Rolls Royce
- Shell

You can find the data for these in the `assets` folder. To add your own assets, you can visit [here](https://finance.yahoo.com/quote/%5EFTSE/components?p=%5EFTSE) to see the FTSE 100 stocks. You can select a stock, navigate to 'Historical Data', then make the time frame for the data a year, and download it in .csv format. Then you can just add it into the `assets` folder and it will work automatically.

The code works by forming a discrete search space of the different weightings and brute forcing the optimal weightings. It also outputs the other weighting options in the form of a graph and shows the optimal postion that has been selected, as you can see below.

![risk v return ](https://github.com/oranbramble/Portfolio-Optimisation/assets/56357864/95b53d6b-a472-47a8-a42c-521531b48774)

This Risk vs Return graph shows the return of an average portfolio (**red** dot), with equal investment in each asset. It also shows the optimal portfolio for the same risk level as the avergae portfolio (**black** dot). This is based on the training data; a graph for the test data will be shown as this graph is closed. The code also produces a covariance matrix and a Returns vs Time graph. 

## Method

To optimise the portfolios, this program uses a simple methodology. It discretises the search space and brute force checks for the optimal weight combination. A weight combination is a weighting of investment in each asset. For example, if you have 5 assets, a weight combination of equal investment in each asset would be [0.2, 0.2, 0.2, 0.2, 0.2]. This means each assets receives 0.2 of the portfolio's total investment equity. 

So, this program creates a list of all possible combinations of the weight vector, assuming each weight must be changed in 0.02 steps. This ensures the program does not run for too long a time period whilst also maintaining a good level of accuracy. Each weight combination is then tested against the test data to see which provides the best expected returns. The weight combination with the highest expected return is then selected as the optimal weight vector.

## How to Run

This has been fully programmed in Python 3.11. To run this, you will need to have this installed. Download the latest version of python [here](https://www.python.org/downloads/).

First, clone this repository to your Command Line Interface's current directory

```
git clone https://github.com/oranbramble/Portfolio-Optimisation.git
```

Then install both `numpy` and `matplotlib` libraries using the `pip` commmand

```
python -m pip install numpy
```
```
python -m pip install matplotlib
```

