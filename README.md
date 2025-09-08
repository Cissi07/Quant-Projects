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
