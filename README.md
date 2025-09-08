# MACD Oscillator

MACD refers to Moving Average Convergence/Divergence. It is a momentum trading strategy which holds the belief that upward/downward momentum has more impact on short term moving average than long term moving average. It only takes 5 minutes for any bloke with no background in finance to trade with MACD signals. Regarding the simplicity of MACD oscillator, it is the most common strategy among the non-professionals in the market. In behavioral economics, the more people believe in the strategy, the more effective the strategy becomes (not always true, e.g. 2008). Therefore, we should not underestimate the power of MACD oscillator.

### Simple Moving Average (SMA):
- SMAs are simpler but lag more than EMAs(exponential moving average) used in MACD
- SMAs generate binary crossover signals
- SMAs are bets for long-term trend identification (e.g. 50/200-day SMAs)
- .rolling().mean() is the standard SMA formula

$SMA_t=\frac{1}{N} \sum^{N-1}_{i=0}x_{t-i}$
e.g. Close = [100, 101, 102, 103, 104]:
- min_periods=3: results = [NaN, NaN, 101.0, 102.0, 103.0]
- min_periods=1: results = [100.0, 100.5, 101.0, 102.0, 103.0]

### Exponential Moving Average (EMA)
- Weights recent prices more than older ones, with weights that decay geometrically

$EMA_t=\alpha x_t+(1-\alpha)EMA_{t-1}$ 
where $x_t$ is the Close at time t and $\alpha=\frac{2}{span+1}$

### Add volatility filer
- Computes a rolling standard deviation of the Close price over the last window rows (trailing window)
- Volatility threshold: Takes a rolling median of that volatility series (again over window rows) and scales it by multiplier. This is your volatility threshold: takes a rolling median of that volatility series and scales it by multiplier
- Only act when current volatility is above the rolling-median threshold


# Pair-trading

## Use Engle-Granger two-step method to test cointegration
The latest statsmodels package should've included johansen test which is more common.
Check sm.tsa.var.vecm.coint_johansen,
the malaise of two-step is the order of the cointegration,
unlike johansen test, two-step method can only detect the first order,
check the following material for further details
https://warwick.ac.uk/fac/soc/economics/staff/gboero/personal/hand2_cointeg.pdf

**Step 1:Look for pairs (or groups) of stocks whose combination is stationary, i.e., they are cointegrated.**

- Perform an OLS regression of one non-stationary variable on another: 
$y_t=\beta_0+\beta_1 x_t+\epsilon_t$
where $\epsilon_t$ is the residual,and is a linear combination of $y_t$ and $x_t$

- Perform Augmented Dickey Fuller (ADF) test: 
$y_t= M + \phi_1 y_{t-1} + \epsilon_t$
Null hypothesis ($H_0$): the series has a unit root ($\phi_1$ = 1, non-stationary); $H_1$: The series is stationary

- When the spread $\epsilon$ is stationary (mean-reverting), whenever the spread deviates (positive or negative $\epsilon$), it will eventually come back

**Step 2: Estimate the Error Correction Model (ECM)**

$$
\underbrace{A(L)\,\Delta y_t}_{\text{lag polynomial of dependent var.}} =
\underbrace{\gamma}_{\text{intercept}} +
\underbrace{B(L)\,\Delta x_t}_{\text{short-run dynamics of explanatory var.}} +
\underbrace{\alpha\,\hat{\varepsilon}_{t-1}}_{\substack{\text{error correction term:}\\ \text{lagged residual from cointegration}}} +
\underbrace{\nu_t}_{\text{error term}}
$$


#### $\alpha$ must be negative:

Suppose yesterday's spread was positive ($\hat{\varepsilon}_{t-1} > 0$):
- which means that $Y$ was too high relative to equilibrium
- If the model is stable, today's change $\Delta Y_t$ should be negative (a downward correction) 
- For this to happen, the adjustment term $\alpha \hat{\varepsilon}_ {t-1}$ must be negative when $\hat{\varepsilon}_{t-1} > 0$.  
- This requires $\alpha < 0$

Likewise, if yesterday's spread was negative ($\hat{\varepsilon}_{t-1} < 0$):
- $Y$ was too low relative to equilibrium, and so today's $\Delta Y_t$ should be positive (an upward correction).  
- Again, this only happens if $\alpha < 0$.


## Signal generation
- First we verify the status of cointegration by checking historical datasets.
- Bandwidth determines the number of data points for consideration, bandwidth is 250 by default, around one year's data points
- If the status is valid, we check the signals
- When z stat gets above the upper bound, we long the bearish one and short the bullish one, vice versa
