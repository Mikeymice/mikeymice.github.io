---
layout: post
title: Linear Regression
date: 2018-11-06T14:37:44.000Z
categories: Data Science Regression Linear Relationship
---

<img src="/images/apples.jpg" class="fit image">
<sup>[Photo: Pixabay](https://cdn.pixabay.com/photo/2017/10/03/10/43/apples-2811968_960_720.jpg)</sup>
## Linear Relationship

Imagine that you are buying apples at a market. The _price point_ of the apples is $5.00 per lbs. You pick up 1.5 lbs and pay 7.5 dollars. The cost depends on the _quantity in weight_. Let's assume you buy apples in different quantities ( 2, 3, ... 10 lbs) at the same _price point_ and record all the costs and quantities. Let's plot the costs and quantities of the apples as the figure below.

![dummy_data](/images/simple_cost.png)

You can see that the dots form a _straight line_. This relation between the cost and the quantity is an example of **linear relationship**.

We can express the line in the equation such as

[![](https://latex.codecogs.com/gif.latex?y&space;=&space;a&space;\times&space;x&space;+&space;b "y = a \times x + b")](https://www.codecogs.com/eqnedit.php?latex=y&space;=&space;a&space;\times&space;x&space;+&space;b)

or

[![](https://latex.codecogs.com/gif.latex?$y=&space;slope&space;\times&space;x&space;+&space;intercept$ "$y= slope \times x + intercept$")](https://www.codecogs.com/eqnedit.php?latex=$y=&space;slope&space;\times&space;x&space;+&space;intercept$)

We can use the following formula to express the line in our graph

[![](https://latex.codecogs.com/gif.latex?TotalCostOfApples&space;=&space;PricePoint&space;\times&space;Quantity&space;+&space;UnderlineCost "TotalCostOfApples = PricePoint \times Quantity + UnderlineCost")](https://www.codecogs.com/eqnedit.php?latex=TotalCostOfApples&space;=&space;PricePoint&space;\times&space;Quantity&space;+&space;UnderlineCost)

For now, we will assume _UnderlineCost_ is 2 because the stores charge you for bags and they are expensive bags! The amount of apples you are buying is the **independent variable** because you _control_ the amount, and it does _not depend_ on other variables. The total cost of apple is the **dependent variable** because the cost _depends_ on the amount you choose. The price point ($/lbs) of the apples is the **coefficient** or **slope** of the independent variable.

Imagine you want to predict the cost of 30 lbs of apple at the same price point. If you look at the history of the costs and quantities, you have no idea how much 30 lbs of apples will cost because we do not have data on it!

![dummy_data](/images/simple_cost_unknown.png)

However, you know that there is a linear relationship between the cost and the quantities of the apples. So you connect all the existing dots and extend the line, the red star represents the cost of 30 lbs of apple will be on the line!

![dummy_data](/images/simple_cost_prediction.png)

We can also calculate the cost with our formula

[![](https://latex.codecogs.com/gif.latex?TotalCostOfApples&space;=&space;5&space;dollars/lbs&space;\times&space;30&space;lbs&space;+&space;2 "TotalCostOfApples = 5 dollars/lbs \times 30 lbs + 2")](https://www.codecogs.com/eqnedit.php?latex=TotalCostOfApples&space;=&space;5&space;dollars/lbs&space;\times&space;30&space;lbs&space;+&space;2)

[![](https://latex.codecogs.com/gif.latex?TtoalCostOfApples&space;=&space;152&space;dollars "TtoalCostOfApples = 152 dollars")](https://www.codecogs.com/eqnedit.php?latex=TtoalCostOfApples&space;=&space;152&space;dollars)

## Linear Regression

You have been keeping track of how much you spend on apples each time you go to grocery stores. The price point can vary. You might buy apples at different grocery stores. You might purchase apples at different times of the year. The price of the apples can also vary based on inflation. Over the years, you have collected a large dataset of the costs and the quantities.

One day you decide to open a pie shop and plan to sell apple pies. You have to purchase apples by the quantities that are larger than you normally buy. You realize that you need to figure out a budget for the apples. You ask yourself the question

> _"What is the likely cost of 180 lbs of apples for the next month?"_

You suddenly remember you have the dataset. You might be able to predict the cost from all the past data. All you need to do is to get a straight line from all the data points. However, after you plot all the data you have collected, you realize that you cannot connect all the dots to form a straight line.

![dummy_data](/images/past_data.png)

You have to take the best guess to fit the line into the figure to explain the linear relationship between the quantities and the costs. This is an example of **linear regression**.

Linear regression is an approach to describe the linear relationship between dependent variable and independent variable(s). The case when there is only one independent variable is called **simple linear regression**. **Multiple linear regression** is when there are more than one independent variable. Imagine your shop is selling not only apple pies but also peach pies. You need to predict the cost of the fruits.

## Prediction From Linear Regression

Linear Regression usually adopt the **least square** approach to describe the relationship. Least square approach minimizes the square of the differences between the predicted values and measured values. In this case, the values are the costs of the apples.

We can use `scipy` library in python to achieve linear regression.

Let's try to predict the likely cost of 180 lbs apples from our dataset.

To demonstrate linear regression, we are going to import necessary libraries and produce some dummy data.

```python
from scipy import stats
import matplotlib.pyplot as plt #for plotting
```

Let's look at our plot again.

![dummy_data](/images/past_data.png)

As you can see you cannot form a straight line by connecting all the dots. Let's try to find the best fitted line with linear regression. `quantities_purchased` is an list of the quantities of the apples you have purchased in the past. `costs` is an list of the costs you paid for the apples. Both lists share the same index. For example, `quantities_purchased[0]` and `costs[0]` describe the same single event when you purchased apples.

```python
reg = stats.linregress(quantities_purchased, costs)
price_point = reg.slope
underline_cost = reg.intercept
print(price_point)
print(underline_cost)
```

> 4.933814751192114<br>
> 2.0781993693435297

We can plot the line with the formula, [![](https://latex.codecogs.com/gif.latex?$y=&space;slope&space;\times&space;x&space;+&space;intercept$ "$y= slope \times x + intercept$")](https://www.codecogs.com/eqnedit.php?latex=$y=&space;slope&space;\times&space;x&space;+&space;intercept$)

![dummy_data](/images/linear_reg.png)

Use our equation

[![](https://latex.codecogs.com/gif.latex?TotalCostOfApples&space;=&space;PricePoint&space;\times&space;Quantity&space;+&space;UnderlineCost "TotalCostOfApples = PricePoint \times Quantity + UnderlineCost")](https://www.codecogs.com/eqnedit.php?latex=TotalCostOfApples&space;=&space;PricePoint&space;\times&space;Quantity&space;+&space;UnderlineCost)

We can conclude that the most likely _PricePoint_ for the apples is `4.933814751192114` (\\$/lbs). We also have `2.0781993693435297` of _UnderlineCost_. Now we can apply 180 to the _Quantity_

```python
budget = price_point*180 + underline_cost
print(budget)
```

> 890.164854583924

We have predicted the expected cost of 180 lbs of apples for your pie store via linear regression!

## References

-   <https://en.wikipedia.org/wiki/Linear_regression>
-   <https://en.wikipedia.org/wiki/Linearity>
