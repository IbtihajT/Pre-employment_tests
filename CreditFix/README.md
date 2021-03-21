# Data Analysis Exercise

Howdy!

The main goal of this exercise is to test your data visualization and data analysis abilities. This exercise shouldn't take more than 5-6 hours to complete. We encourage you to submit your solution even if you are not able to complete the exercise in given time. Please upload your solution on a code sharing tool such as github, gitlab or bitbucket.

## Exercise 

### Introduction

Financial institutions incur significant losses due to the default of vehicle loans. This has led to the tightening up of vehicle loan underwriting and increased vehicle loan rejection rates. The need for a better credit risk scoring model is also raised by these institutions. This warrants a study to estimate the determinants of vehicle loan default. 

The data for this exercise is available in the `data/` folder.

The file `data.csv` in the `data/` folder contains the information of loan as well as the loanee. Following information regarding the loan and loanee are provided:
-	Loanee Information (Demographic data like age, income, Identity proof etc.)
-	Loan Information (Disbursal details, amount, EMI, loan to value ratio etc.)
-	Bureau data & history (Bureau score, number of active accounts, the status of other loans, credit history etc.)

The details of the numerous fields in the csv file are explained in the microsoft word file: `Data Dictionary.xlsx` in `data/` folder


### Expectations

First of all, you will identify fields with more than 50% null data. These fields should be removed from analysis. 

There is a field `LOAN_DEFAULT` in the data.csv file. This field tells us whether the loanee defaulted his/her loan or not. This is our target variable. We should be able to predict, using the available data, whether the loanee will default his/her loan or not.

You should also be able to deduce new field in the data; such as age of the loanee based on the Date of birth. 

Given below are your tasks for this exercise: 

1.	Analyze, using graphs, how different fields (loan info and loanee info) correlate to a loanee defaulting his/her loan or not.
 
2.	Analyze, using graphs, whether the demographic data plays a major role in a loanee defaulting his/her loan or the financial data.

3.	Present the fields that should be used in the future analysis.

4.	Present your recommendations for the estimators that can be used as a model to predict loan defaults.



You are expected to submit your solution in a python3 notebook in jupyter. Pandas is the python data analysis library that must be used for data analysis. Data visualization should be done using matplotlib, or any other library of your choice.
You will be judged on your visualization approach, code quality and accuracy of analysis. Feel free to contact us anytime if you have any queries regarding the exercise.

Thank you and Best of luck! 
