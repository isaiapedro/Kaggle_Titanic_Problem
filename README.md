# Titanic Problem on Kaggle.com!
### (Predictive Model for Discrete Data)

## Contents

- [Introduction](#introduction)
- [Data Overview](#data-overview)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

This repository was made to solve the *Titanic Problem* on [Kaggle.com](https://www.kaggle.com/c/titanic).

<br/>

**A brief explanation of the problem:**

> The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

<br/>

This work was a collaboration between me and @descarteslover and the objective was to complete the challenge with the best precision possible and take a first hands-on look into ML models and Kaggle competitions.

## Data Overview

First, we looked at the data that was given by the .csv files “Train” and “Test”.

Index | PassengerId | Survived |	Pclass |	Name |	Sex |	Age | SibSp | Parch | Ticket | Fare |	Cabin | Embarked
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |--- | ---
0 | 1 | 0.0 | 3 | Braund, Mr. Owen Harris | male | 22.0 | 1 | 0 | A/5 | 21171 | 7.2500 | NaN | S

<br>

We noticed that there’s a lot of missing values and some data that could be grouped, specifically the parents and sibling’s columns.


<br>

Category | Null Values
--- | ---
PassengerId | 0
Survived |	418
Pclass | 0
Name | 0
Sex |	0
Age | 263
SibSp | 0
Parch | 0
Ticket | 0
Fare |	1
Cabin | 1014
Embarked | 2

<br>

For the missing values, a correlation matrix was plotted, and it was possible to see the similar values between each column. For example, the closest value to the **Age** column was **Pclass**, so we filled the null values of **Age** based on the mean age of the **Pclass group** in which the passenger belonged.

<br>

```
for i in titanic_df.index:
    if pd.isnull(titanic_df.loc[i, 'Age']):
        if titanic_df['Pclass'][i] == 1:
            titanic_df.loc[i, 'Age'] = titanic_df[titanic_df['Pclass'] == 1]['Age'].mean()
        elif titanic_df['Pclass'][i] == 2:
            titanic_df.loc[i, 'Age'] = titanic_df[titanic_df['Pclass'] == 2]['Age'].mean()
        elif titanic_df['Pclass'][i] == 3:
            titanic_df.loc[i, 'Age'] = titanic_df[titanic_df['Pclass'] == 3]['Age'].mean()
    else:
        continue
```

![](https://github.com/isaiapedro/Kaggle_Titanic_Problem/blob/pedro/correlation-matrix.png?raw=true)

<br>

For the missing values of **Cabin**, we realized that most passengers that falled to that category did't survived, so we divided them between *True* (Cabin Assigned) and *False* statements.

<br>


![](https://github.com/isaiapedro/Kaggle_Titanic_Problem/blob/pedro/cabin-assigned.png?raw=true)

<br>

For the grouping of values, we used the command `qcut` to split the values in equal distributions, and then assigned each distribution with values from 0 to n-1.

```
titanic_df['Fare_Cut'] = pd.qcut(titanic_df['Fare'], 6)

titanic_df.loc[titanic_df['Fare'] <= 7.775, 'Fare'] = 0
titanic_df.loc[(titanic_df['Fare'] > 7.775) & (titanic_df['Fare'] <= 8.662), 'Fare'] = 1
titanic_df.loc[(titanic_df['Fare'] > 8.662) & (titanic_df['Fare'] <= 14.454), 'Fare'] = 2
titanic_df.loc[(titanic_df['Fare'] > 14.454) & (titanic_df['Fare'] <= 26), 'Fare'] = 3
titanic_df.loc[(titanic_df['Fare'] > 26) & (titanic_df['Fare'] <= 53.1), 'Fare'] = 4
titanic_df.loc[(titanic_df['Fare'] > 53.1) & (titanic_df['Fare'] <= 512.329), 'Fare'] = 5
titanic_df.loc[(titanic_df['Fare'] > 512.329), 'Fare']
```

<br>

The result is equally spaced ordinal values that can be accessed by the training model through the ***Ordinal Encoder*** tool.

<br>

![](https://github.com/isaiapedro/Kaggle_Titanic_Problem/blob/pedro/fare-cut.png?raw=true)

<br>

## Results

### Conclusion

Thanks for reading up until here. We had a ton of fun doing this notebook and got a lot of useful insights on data manipulation and the implementation of some models, as Gradiant Boosting, Random Forest, Logistic Regression and Extra Tree.

If you want to see more Kaggle solutions, see the Flower Classification Problem or go to my github page. Feel free to reach me on [LinkedIn](https://www.linkedin.com/in/isaiapedro/) or my [Webpage](https://github.com/isaiapedro/Portfolio-Website).

Bye!
