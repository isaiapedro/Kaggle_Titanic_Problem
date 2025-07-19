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

For the missing values of **Cabin**, we realized that most passengers that fall to that category did't survived, so we divided them between *True* (Cabin Assigned) and *False* statements.

<br>


![](https://github.com/isaiapedro/Kaggle_Titanic_Problem/blob/pedro/cabin-assigned.png?raw=true)


<br>


For the grouping of values, we used the command `qcut` to split the values in equal distributions, and then assigned each distribution with values from 0 to n-1.

<br>

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

The sample values are equally spaced and ordenated. It can be accessed by the training model through the ***Ordinal Encoder*** tool.

<br>

![](https://github.com/isaiapedro/Kaggle_Titanic_Problem/blob/pedro/fare-cut.png?raw=true)

<br>

## Results

For the training part, we used the **Column Transformer** feature of *Sklearn* to transform the data with the help of **OneHot Encoder** and **Ordinal Encoder**. The result is a single feature space, with homogeneous data of binary-divided columns and ascending ordinal values.

<br>

```
ct = make_column_transformer(
    (ohe, ['Sex', 'Embarked', 'Cabin_Assigned', 'Pclass']),
    (ode, ['FamilySize_Grouped', 'TicketNumber_Grouped']),
    remainder = 'passthrough'
)
```
<br>

We tracked the result of the top 4 models that were implemented and the speed in which each one was trained:

<br>

Model | Efficiency | Speed
--- | --- | ---
Gradient Boosting | 0.78 | 19 m 0.8 s
Random Forest |	0.77 | 24 m 37 s
Logistic Regression | 0.77 | 0.7 s
Extra Trees | 0.77 | 1 m 26 s

<br>

It's noticeable the difference of speed that some of the algorithms run compared to the others. Besides that, the accuracy of prediction is very similar.

After finding the real output of the challenge, we can compare it to our solution to see what is the profile of those participants that we wrongfully predicted and if there're any correlations to these values and the fact that our models couldn't predict their true outcome.

<br>

![](https://github.com/isaiapedro/Kaggle_Titanic_Problem/blob/pedro/sex-correlation.png?raw=true)

<br>

![](https://github.com/isaiapedro/Kaggle_Titanic_Problem/blob/pedro/familysize-grouped-correlation.png?raw=true)

<br>

We can see that most of the passengers that we predicted wrongfully were men that survived and also people that were not accompanied by family members and still survived. Those two groups of people were part of a minority, since in our analysis, the vast majority of those people did not survived.

<br>

If we were to improve the model above, the next step would be to improve the dataset in a way that showcases these minor variances of each divided group.

<br>

## Conclusion

Thanks for reading up until here. We had a ton of fun doing this notebook and got a lot of useful insights on data manipulation and the implementation of some ML models, such as Gradiant Boosting, Random Forest, Logistic Regression and Extra Tree.

If you want to see more Kaggle solutions, see the Flower Classification Problem or go to my github page. Feel free to reach me on [LinkedIn](https://www.linkedin.com/in/isaiapedro/) or my [Webpage](https://github.com/isaiapedro/Portfolio-Website).

Bye!
