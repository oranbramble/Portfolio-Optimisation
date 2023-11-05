# Portfolio-Optimisation


This script takes any number of 'assets' and finds the optimal weighting of investment on each stock to provide maximum returns. Each 'asset' is a stock of a company that is accessible on the stock market. For convenience, I have already included 5 assets so that this code can be run immediately. These 5 assets are:

- Auto Trader
- Experian
- Rightmove
- Rolls Royce
- Shell

You can find the data for these in the `assets` folder. To add your own assets, you can visit [here](https://finance.yahoo.com/quote/%5EFTSE/components?p=%5EFTSE) to see the FTSE 100 stocks. You can select a stock, navigate to 'Historical Data', then make the time frame for the data a year, and download it in .csv format. Then you can just add it into the `assets` folder and it will work automatically.

The code works by forming a discrete search space of the different weightings and brute forcing the optimal weightings. It also outputs the other weighting options in the form of a graph and shows the optimal postion that has been selected, as you can see below.


