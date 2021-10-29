import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


col_list = ['#005f9a', '#00CDCD', '#f1bdbf']
sns.set_palette(col_list)

credit_data = pd.read_csv("Credit_N400_p9.csv", index_col=0)

# Converting to the appropriate data type
credit_data.Gender = credit_data.Gender.astype('category')
credit_data.Student = credit_data.Student.astype('category')
credit_data.Married = credit_data.Married.astype('category')

# print the dataset with numerical values / integers
print(credit_data.describe())
print(credit_data.head())


# print the dataset with string dataset (that has been converted) 
print(credit_data.describe(include=['category']))


credit_data['Positive'] = np.where(credit_data['Balance']>0, 'Yes', 'No')  
print(credit_data.Positive.describe())

# plotting the dataset for the credit card balance
sns.distplot(credit_data.Balance)
plt.show()

# finding whether the credit card has positive balance
# The print only the positive credit balance
pos_credit_data = credit_data.loc[credit_data.Balance>0,].copy()
print(pos_credit_data.Balance.describe())

# ploting positive credit balance 
sns.displot(pos_credit_data.Balance)
plt.show()

numeric_credit_data = credit_data.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(8,8))
plt.matshow(credit_data.corr(), cmap=plt.cm.Blues, fignum=1)
plt.colorbar()
tick_marks = [i for i in range(len(numeric_credit_data.columns))]
plt.xticks(tick_marks, numeric_credit_data.columns)
plt.yticks(tick_marks, numeric_credit_data.columns)
plt.show()

# Performing Correlation coefficient using pearsonr
from scipy.stats import pearsonr

r1, p1 = pearsonr(credit_data.Balance, credit_data.Limit)
msg = "Correlation coefficient Balance-Limit: {}\n p-value: {}\n"
print(msg.format(r1, p1))
r2, p2 = pearsonr(credit_data.Balance, credit_data.Rating)
msg = "Correlation coefficient Balance-Rating: {}\n p-value: {}\n"
print(msg.format(r2, p2))
r3, p3 = pearsonr(credit_data.Balance, credit_data.Income)
msg = "Correlation coefficient Balance-Income: {}\n p-value: {}\n"
print(msg.format(r3, p3))
r4, p4 = pearsonr(credit_data.Limit, credit_data.Rating)
msg = "Correlation coefficient Limit-Rating: {}\n p-value: {}\n"
print(msg.format(r4, p4))
r5, p5 = pearsonr(credit_data.Limit, credit_data.Income)
msg = "Correlation coefficient Limit-Income: {}\n p-value: {}\n"
print(msg.format(r5, p5))
r6, p6 = pearsonr(credit_data.Rating, credit_data.Income)
msg = "Correlation coefficient Rating-Income: {}\n p-value: {}\n"
print(msg.format(r6, p6))

#Finding the correlation of the Credit card limit with Credit card Rating
sns.regplot(x='Limit',
           y='Rating',
           data=credit_data,
           scatter_kws={'alpha':0.2},
           line_kws={'color':'black'})
plt.show()

# Examining the Categorical variable ( Married, Gender,Student) and their Relationship to the credit card balance
f, axes = plt.subplots(2, 2, figsize=(15, 6))
f.subplots_adjust(hspace=.3, wspace=.25)
credit_data.groupby('Gender').Balance.plot(kind='kde', ax=axes[0][0], legend=True, title='Balance by Gender')
credit_data.groupby('Student').Balance.plot(kind='kde', ax=axes[0][1], legend=True, title='Balance by Student')
credit_data.groupby('Married').Balance.plot(kind='kde', ax=axes[1][0], legend=True, title='Balance by Married')
plt.show()

# Student appear to be associated with the Balance
# therefore we closely examine it by plotting boxplot
sns.boxplot(x='Student', y='Balance', data = credit_data)
plt.show()

# Perform Regression using The Ordinary Least Square algorithm
# For the p = 9 features
mod0 = smf.ols('Balance ~ Income + Rating + Cards + Age + Education + Gender + Student + Married', data = credit_data).fit()
print(mod0.summary())

# Interaction with the features ~ Rating and Income
mod3 = smf.ols('Balance ~ Income + I(Income**2) + Age + Student + Income*Rating', data = credit_data).fit()
print(mod3.summary())

# plotting the values of Age and Married
sns.lmplot(x="Age", 
           y="Balance", 
           hue="Married", 
           ci=None,
           data=pos_credit_data);
plt.show()

# Finding the best fit - model
# Trying polynomial relationship btn the income and the Balance
# Interaction between the Income and Rating

active_mod7 = smf.ols('Balance ~ Income + I(Income**2) + Rating + Age + Student + Income*Rating', 
                      data = pos_credit_data).fit()
print(active_mod7.summary())

# Using Logistic regression
# to determine the factors which influence or determine 
# whether the card owner has positive credit balance

log_mod = smf.glm('Positive ~ Limit + Rating + Income + Age + Cards + Education', 
                   data = credit_data,
                   family=sm.families.Binomial()).fit()
print(log_mod.summary())



# Making the model Prediction 

data_new=pd.DataFrame({'Income':np.random.normal(45, 20, 40),
                    'Rating':np.random.normal(355, 55, 40),
                    'Limit':np.random.normal(4735, 200, 40),
                    'Age':np.random.normal(56, 17, 40),
                    'Cards':list(range(0,10))*4,
                    'Student':['Yes']*20+['No']*20})
data_new.Cards[data_new.Cards == 0] = 3
data_new.Income[data_new.Income <= 0] = data_new.Income.mean()
data_new.Rating[data_new.Rating <= 0] = data_new.Rating.mean()
data_new.Limit[data_new.Limit <= 0] = data_new.Limit.mean()
data_new['Balance']= active_mod7.predict(data_new)
print(data_new.describe())

mod8 = smf.ols('Balance ~ Income + I(Income**2) + Age + Student + Income*Rating + Limit + Cards', data = credit_data).fit()
print(mod8.summary())