---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 1.0.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## Introduction

There are many powerful tools and packages in **R**. I have been spoiled by the convenient tools for interpretation. For example, the `lm` package provides readable summary output and easy to use formula as input for the model. However, things are not as easy and out of box in the `python` world. Although the machine learning packages such as `sklearn` and `tensoflow` are mature, things can be tricky when we deal with linear regression and categorical variable.

Recently, I have opportunity to work with a company for my capstone project. The company has strong preference for python language. One of the main objectives of the project is interpretibility. The golden rule of interpretibility is to avoid the black box machine learning model such as neural network. Our group decide to use logistic regression for our project. Being familiar with *R* package and the conveniency of use of formula and under-the-hood of handling of categorical variables, we were surprised of how complicated things can be with python. Given that pandas pacakge come with its `get_dummy` function, it comes with the imfamous [dummy trap](https://www.algosome.com/articles/dummy-variable-trap-regression.html).

Thanksfully, [Statsmodel](https://www.statsmodels.org/stable/index.html) comes in for the rescue. Statsmodel is a Python module that provides classes and functions for the estimation of many different statistical models. It also support *R*-style formula. 


```python
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
```

```python
# load data 
dat = sm.datasets.get_rdataset("Guerry", "HistData").data
```

Let's take a pick at our data!

```python
dat.head()
```

We can use `R`-style formula with the new api from stastsmodel!

```python
results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()
```

One of the great advantage of statsmodel is that it provide readable summary report for the models. 

```python
print(results.summary())
```

However, there ares still some short coming from the package. For example, the current implementation of **Logistic Regression** does not support weight for the variables. Although **glm** does have `freq_weight` and other parameters, they are not well tested and documented. This can be troublesome when we are dealing with imbalanced data. 


A good way to solve this problem is to use **sklearn** package. What if we can use R-style formula with sklearn?

The answer is **YES, WE CAN**



```python

```

```python

```

under the hood of statsmodel, the formula is handled by the patsy, which is the package which handles the formula. 

We can actually use patsy api to return the dataframe X and y for the sklearn model from our data


The function we will be using is [dmatrices](https://patsy.readthedocs.io/en/latest/API-reference.html#patsy.dmatrices). We will use "dataframe" for our `return_type`

```python
from patsy import dmatrices, dmatrix, demo_data

y, X = dmatrices("Lottery ~ Literacy + np.log(Pop1831)", data = dat, return_type= "dataframe")
```

```python
X.head()
```

```python
y.head()
```

The beauty of the patsy and the formula is that it is much more readable and it can support dummy encoding right from the formula. Let's say we are adding the categorical variable,`department` in our formula and all we need to do is to enclose Department with `C()` (**C** stands for categorical)

```python
y, X = dmatrices("Lottery ~ Literacy + np.log(Pop1831) +  C(Department)", data = dat, return_type= "dataframe")
```

```python
X.head()
```

```python
list(X.columns.values)
```

By default, the `C(Department)` is the same as `C(Department, Treatment)` which is the dummy encoding for categorical variables and by default it will drop the first value to avoid dummy trap. Patsy also support effect encoding which we will go over in future post. 


Another advantage of patsy formula is it also support various effect coding for the categorical variable


```python

```
