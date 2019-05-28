---
layout: post
title: Regression Model With Formula in Python
date: 2019-05-20T14:37:44.000Z
categories:
  - package linear-model python patsy statsmodel python formula
---

# R-style Formula in Python!?

There are many powerful tools and packages in **R**. I have been spoiled by the convenient tools for interpretation of coefficients. For example, the `lm` package provides readable summary output and easy to use formula as input for the model. However, things are not as easy in the `python` world. Although the machine learning packages such as `sklearn` and `tensoflow` are mature, handling categorical variable can be complicated.

Recently, I have opportunity to work with a company for my capstone project. The company has strong preference for python language. One of the main objectives of the project is interpretation of the data. The golden rule of interpretation is to avoid the black box machine learning models such as neural network. Our group decided to use logistic regression for our project. Being familiar with _R_ package and the convenience of use of formula and under-the-hood of handling of categorical variables, we were surprised of how complicated things can be with python. Given that pandas package come with its `get_dummy` function, it comes with the imfamous [dummy trap](https://www.algosome.com/articles/dummy-variable-trap-regression.html).

Thankfully, [Statsmodel](https://www.statsmodels.org/stable/index.html) comes in for the rescue. Statsmodel is a Python module that provides classes and functions for the estimation of many different statistical models. It also support _R_-style formula.


![](/images/patsy_post/import.png)



![](/images/patsy_post/data_import.png)

Let's take a pick at our data!


<div>
  <style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
  <table border="1" class="dataframe">
  <thead><tr style="text-align: right;"><th></th><th>dept</th><th>Region</th><th>Department</th><th>Crime_pers</th><th>Crime_prop</th><th>Literacy</th><th>Donations</th><th>Infants</th><th>Suicides</th><th>MainCity</th><th>...</th><th>Crime_parents</th><th>Infanticide</th><th>Donation_clergy</th><th>Lottery</th><th>Desertion</th><th>Instruction</th><th>Prostitutes</th><th>Distance</th><th>Area</th><th>Pop1831</th></tr></thead>
  <tbody><tr><th>0</th><td>1</td><td>E</td><td>Ain</td><td>28870</td><td>15890</td><td>37</td><td>5098</td><td>33120</td><td>35039</td><td>2:Med</td><td>...</td><td>71</td><td>60</td><td>69</td><td>41</td><td>55</td><td>46</td><td>13</td><td>218.372</td><td>5762</td><td>346.03</td></tr><tr><th>1</th><td>2</td><td>N</td><td>Aisne</td><td>26226</td><td>5521</td><td>51</td><td>8901</td><td>14572</td><td>12831</td><td>2:Med</td><td>...</td><td>4</td><td>82</td><td>36</td><td>38</td><td>82</td><td>24</td><td>327</td><td>65.945</td><td>7369</td><td>513.00</td></tr><tr><th>2</th><td>3</td><td>C</td><td>Allier</td><td>26747</td><td>7925</td><td>13</td><td>10973</td><td>17044</td><td>114121</td><td>2:Med</td><td>...</td><td>46</td><td>42</td><td>76</td><td>66</td><td>16</td><td>85</td><td>34</td><td>161.927</td><td>7340</td><td>298.26</td></tr><tr><th>3</th><td>4</td><td>E</td><td>Basses-Alpes</td><td>12935</td><td>7289</td><td>46</td><td>2733</td><td>23018</td><td>14238</td><td>1:Sm</td><td>...</td><td>70</td><td>12</td><td>37</td><td>80</td><td>32</td><td>29</td><td>2</td><td>351.399</td><td>6925</td><td>155.90</td></tr><tr><th>4</th><td>5</td><td>E</td><td>Hautes-Alpes</td><td>17488</td><td>8174</td><td>69</td><td>6962</td><td>23076</td><td>16171</td><td>1:Sm</td><td>...</td><td>22</td><td>23</td><td>64</td><td>79</td><td>35</td><td>7</td><td>1</td><td>320.280</td><td>5549</td><td>129.10</td></tr></tbody>
</table>
  <p>5 rows × 23 columns</p>
</div>

We can use `R`-style formula with the new api from stastsmodel!

![](/images/patsy_post/summary.png)

One of the great advantage of statsmodel is that it provide readable summary report for the models.


```
OLS Regression Results
==============================================================================
Dep. Variable:                Lottery   R-squared:                       0.348
Model:                            OLS   Adj. R-squared:                  0.333
Method:                 Least Squares   F-statistic:                     22.20
Date:                Mon, 27 May 2019   Prob (F-statistic):           1.90e-08
Time:                        12:27:54   Log-Likelihood:                -379.82
No. Observations:                  86   AIC:                             765.6
Df Residuals:                      83   BIC:                             773.0
Df Model:                           2
Covariance Type:            nonrobust
===================================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------
Intercept         246.4341     35.233      6.995      0.000     176.358     316.510
Literacy           -0.4889      0.128     -3.832      0.000      -0.743      -0.235
np.log(Pop1831)   -31.3114      5.977     -5.239      0.000     -43.199     -19.424
==============================================================================
Omnibus:                        3.713   Durbin-Watson:                   2.019
Prob(Omnibus):                  0.156   Jarque-Bera (JB):                3.394
Skew:                          -0.487   Prob(JB):                        0.183
Kurtosis:                       3.003   Cond. No.                         702.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

However, there are still some shortcoming from the package. For example, the current implementation of **Logistic Regression** does not support weight for the variables. Although **glm** does have `freq_weight` and other parameters, they are not well tested and documented. This can be troublesome when we are dealing with imbalanced data.

A good way to solve this problem is to use **sklearn** package. What if we can use R-style formula with sklearn?

The answer is **YES, WE CAN**

Under the hood of statsmodel, the formula is handled by the [`patsy`](https://patsy.readthedocs.io/en/latest/) package, which is the package which handles the formula.

We can actually use patsy api to return the dataframe X and y for the sklearn model from our data

The function we will be using is [dmatrices](https://patsy.readthedocs.io/en/latest/API-reference.html#patsy.dmatrices). We will use "dataframe" for our `return_type` parameter.

![](/images/patsy_post/patsy.png)

```python
X.head()
```

<div>
  <style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
  <table border="1" class="dataframe">
  <thead><tr style="text-align: right;"><th></th><th>Intercept</th><th>Literacy</th><th>np.log(Pop1831)</th></tr></thead>
  <tbody><tr><th>0</th><td>1.0</td><td>37.0</td><td>5.846525</td></tr><tr><th>1</th><td>1.0</td><td>51.0</td><td>6.240276</td></tr><tr><th>2</th><td>1.0</td><td>13.0</td><td>5.697966</td></tr><tr><th>3</th><td>1.0</td><td>46.0</td><td>5.049215</td></tr><tr><th>4</th><td>1.0</td><td>69.0</td><td>4.860587</td></tr></tbody>
</table>
</div>

```python
y.head()
```

<div>
  <style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
  <table border="1" class="dataframe">
  <thead><tr style="text-align: right;"><th></th><th>Lottery</th></tr></thead>
  <tbody><tr><th>0</th><td>41.0</td></tr><tr><th>1</th><td>38.0</td></tr><tr><th>2</th><td>66.0</td></tr><tr><th>3</th><td>80.0</td></tr><tr><th>4</th><td>79.0</td></tr></tbody>
</table>
</div>

The beauty of the patsy and the formula are that it is much more readable and it can support dummy encoding right from the formula. Let's say we are adding the categorical variable,`department` in our formula, and all we need to do is to enclose Department with `C()` (**C** stands for categorical)

![](/images/patsy_post/categorical.png)

```python
X.head()
```

<div>
  <style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
  <table border="1" class="dataframe">
  <thead><tr style="text-align: right;"><th></th><th>Intercept</th><th>C(Department)[T.Aisne]</th><th>C(Department)[T.Allier]</th><th>C(Department)[T.Ardeche]</th><th>C(Department)[T.Ardennes]</th><th>C(Department)[T.Ariege]</th><th>C(Department)[T.Aube]</th><th>C(Department)[T.Aude]</th><th>C(Department)[T.Aveyron]</th><th>C(Department)[T.Bas-Rhin]</th><th>...</th><th>C(Department)[T.Tarn]</th><th>C(Department)[T.Tarn-et-Garonne]</th><th>C(Department)[T.Var]</th><th>C(Department)[T.Vaucluse]</th><th>C(Department)[T.Vendee]</th><th>C(Department)[T.Vienne]</th><th>C(Department)[T.Vosges]</th><th>C(Department)[T.Yonne]</th><th>Literacy</th><th>np.log(Pop1831)</th></tr></thead>
  <tbody><tr><th>0</th><td>1.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>...</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>37.0</td><td>5.846525</td></tr><tr><th>1</th><td>1.0</td><td>1.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>...</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>51.0</td><td>6.240276</td></tr><tr><th>2</th><td>1.0</td><td>0.0</td><td>1.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>...</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>13.0</td><td>5.697966</td></tr><tr><th>3</th><td>1.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>...</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>46.0</td><td>5.049215</td></tr><tr><th>4</th><td>1.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>...</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>69.0</td><td>4.860587</td></tr></tbody>
</table>
  <p>5 rows × 88 columns</p>
</div>

```python
# let's print out the columns
list(X.columns.values)
```

```
['Intercept',
 'C(Department)[T.Aisne]',
 'C(Department)[T.Allier]',
 'C(Department)[T.Ardeche]',
 'C(Department)[T.Ardennes]',
 'C(Department)[T.Ariege]',
 'C(Department)[T.Aube]',
 'C(Department)[T.Aude]',
 'C(Department)[T.Aveyron]',
 'C(Department)[T.Bas-Rhin]',
 'C(Department)[T.Basses-Alpes]',
 'C(Department)[T.Basses-Pyrenees]',
 'C(Department)[T.Bouches-du-Rhone]',
 'C(Department)[T.Calvados]',
 'C(Department)[T.Cantal]',
 'C(Department)[T.Charente]',
 'C(Department)[T.Charente-Inferieure]',
 'C(Department)[T.Cher]',
 'C(Department)[T.Correze]',
 'C(Department)[T.Corse]',
 "C(Department)[T.Cote-d'Or]",
 'C(Department)[T.Cotes-du-Nord]',
# omit rest of the columns from `Department` variable
  ...
 'Literacy',
 'np.log(Pop1831)']
```

By default, the `C(Department)` is the same as `C(Department, Treatment)` which is the dummy encoding for categorical variables and by default it will drop the first value to avoid dummy trap.

Another advantage of patsy formula is it also support various effect coding for the categorical variables. We will go over effect coding in the future post. If you are interested, you can check out their website [here](https://www.statsmodels.org/devel/contrasts.html)

`patsy` formula also include interaction between variables.

```
y ~ a*b
```

`a*b` is shorthand for `a + b + a:c`. You can read more on their documentation page [here](https://patsy.readthedocs.io/en/latest/formulas.html)

In summary, having formula for machine learning model can be done in not only **R** but also **python**. The advantages of the having the formula are
- An unevaluated expression
-  Readability
-  Flexible handling of the categorical variables
- Provide interaction for regression models.

References:
- https://www.statsmodels.org/devel/example_formulas.html
- https://www.statsmodels.org/devel/contrasts.html
- https://patsy.readthedocs.io/en/latest/
