# University Name
CBS International Business School

<img src="https://github.com/CypranA/Master-Thesis-Statistical-Analysis/blob/main/cbs_logo.png" alt="CBS Logo" width="200"/>


---

## Master Thesis Topic: 
Leveraging Big Data Analytics for Enhanced Financial Inclusion in Nigeria: A Comprehensive Analysis of Strategies, Impact, And Challenges in Developing Economies.

---

## Program: 
Master of Arts in International Business

---

## Specialization: 
Digital Transformation Management

---

## Supervisor: 
Prof. Dr. Geoffrey Writes

---

### Student Details:
- **Name**: Kosisochukwu Cypran Akubude
- **Student Number**: 1195501031
- **Date**: April 14, 2024


# TABLE OF CONTENTS
1. Summary
2. Phase 1: Ask
3. Phase 2: Prepare
* 3.1. Data Location
* 3.2. Data Organization
* 3.3. Data Credibility
* 3.4. Data License and Privacy
4. Phase 3: Process
5. Phase 4: Analyze
* 5.1. Regression
* 5.2. Pearson Correlation Test
* 5.3. Test for Multicollinearity
* 5.4. Descriptive Statistics
* 5.5. Hypothesis Testing
6. Phase 5: Share
7. Phase 6: Conclusion (Act)

# 1. Summary

The purpose of this study is to analyze the relationship between financial inclusion and big data, including economic data variables. We aim towards analysing the datasets and discuss findings. The following tests will be conducted in this analysis. 

- Regression
- Pearson Correlation Test
- Test for Multicollinearity
- Descriptive Statistics
- Hypothesis Testing


# 2. Phase 1: Ask

Questions guiding our analysis incude:
- What is the expected relationship between financial inclusion and big data?
- Do big data sources have a significant impact on increased access to financial services?
- Does an economy with a structured data, accessible data source, and data infrastructure help streamline digital financial inclusion?

# 3. Phase 2: Prepare
**3.1. Data Location:** All data used in this study were obtained from the World Bank DataBank. Data variables for Model 1 were obtained from the Global Findex Database whilst data variables for Model 2 were obtained from the Statistical Performance Indicator Database.

**3.2. Data Organization:** The extracted datasets from the Global Findex dataset comprised 1,450 rows whilst the SPI dataset extracted has 625 rows. Each dataset were organized by columns such as country names, series or variable description, and the corresponding years for each country. The downloaded data have already been cleaned and orginized in Excel before being imported for analysis with Python. The data used in this study therefore are organized by the coutnries and data variables in different columns.  

**3.3. Data Credibility:** The data used are based on the survey of adults from the age of 15 and above accross all countries. The data were last updated in 2021. Only values for the year 2021 are considered in this analysis due to missing data for the previous year. 

**3.4. Data License and Privacy:** The metadata from the datasets shows that its an open-source data which justifies the use for this study. The licence can be accessed via this link: https://datacatalog.worldbank.org/public-licenses#cc-by
![image.png](attachment:image.png)

# 4. Phase 3: Process

Excel has been used to clean and restructure the dataset in a way that is more convenient for analysis. All variables needed for the study have been selected and placed in column sections. The datasets used in this study are named as follow:

- **model1:** Contains financial inclusion and big data source variables.

- **model2:** Contains financial inclusion and SPI variables.

# 5. Phase 4: Analyze


```python
#importing relevant packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr
```

# Importing Model 1 and Model 2 Datasets


```python
#importing model1 dataframe and indexing country column to analyse on numerical data types
model1 = pd.read_csv("model1.csv")
model2 = pd.read_csv('model2.csv')
display(model1.head())
display(model2.head())
```


<div>
<style scoped>
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
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>internet_access</th>
      <th>mobile_utility_payment</th>
      <th>digital_payment</th>
      <th>mobile_phone_ownership</th>
      <th>mobile_bill_payment</th>
      <th>mobile_money_account</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Algeria</td>
      <td>82.91</td>
      <td>1.27</td>
      <td>33.74</td>
      <td>95.18</td>
      <td>3.57</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Benin</td>
      <td>28.85</td>
      <td>5.31</td>
      <td>43.69</td>
      <td>72.39</td>
      <td>7.17</td>
      <td>36.68</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Botswana</td>
      <td>50.40</td>
      <td>15.81</td>
      <td>51.78</td>
      <td>88.43</td>
      <td>20.28</td>
      <td>36.57</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Burkina Faso</td>
      <td>23.76</td>
      <td>6.23</td>
      <td>33.29</td>
      <td>76.13</td>
      <td>9.83</td>
      <td>24.67</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cameroon</td>
      <td>40.48</td>
      <td>8.60</td>
      <td>49.85</td>
      <td>75.23</td>
      <td>11.59</td>
      <td>42.43</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
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
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>source_data_capacity</th>
      <th>spi_data_use</th>
      <th>spi_data_services</th>
      <th>spi_data_products</th>
      <th>spi_data_sources</th>
      <th>spi_data_infrastructure</th>
      <th>account</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Benin</td>
      <td>60.0</td>
      <td>80.0</td>
      <td>69.90</td>
      <td>83.63</td>
      <td>29.18</td>
      <td>50.0</td>
      <td>48.61</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Botswana</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>67.53</td>
      <td>77.80</td>
      <td>61.53</td>
      <td>40.0</td>
      <td>58.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Burkina Faso</td>
      <td>60.0</td>
      <td>80.0</td>
      <td>69.03</td>
      <td>81.46</td>
      <td>36.66</td>
      <td>60.0</td>
      <td>36.11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cameroon</td>
      <td>20.0</td>
      <td>60.0</td>
      <td>63.87</td>
      <td>82.09</td>
      <td>23.34</td>
      <td>35.0</td>
      <td>51.65</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Chad</td>
      <td>50.0</td>
      <td>70.0</td>
      <td>57.03</td>
      <td>75.84</td>
      <td>18.02</td>
      <td>30.0</td>
      <td>23.65</td>
    </tr>
  </tbody>
</table>
</div>



```python
#checking and confirming data types
model1.info()
print("\n")
model2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 40 entries, 0 to 39
    Data columns (total 7 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   country                 40 non-null     object 
     1   internet_access         40 non-null     float64
     2   mobile_utility_payment  40 non-null     float64
     3   digital_payment         40 non-null     float64
     4   mobile_phone_ownership  40 non-null     float64
     5   mobile_bill_payment     40 non-null     float64
     6   mobile_money_account    40 non-null     float64
    dtypes: float64(6), object(1)
    memory usage: 2.3+ KB
    
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 35 entries, 0 to 34
    Data columns (total 8 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   country                  35 non-null     object 
     1   source_data_capacity     35 non-null     float64
     2   spi_data_use             35 non-null     float64
     3   spi_data_services        35 non-null     float64
     4   spi_data_products        35 non-null     float64
     5   spi_data_sources         35 non-null     float64
     6   spi_data_infrastructure  35 non-null     float64
     7   account                  35 non-null     float64
    dtypes: float64(7), object(1)
    memory usage: 2.3+ KB
    

We can confirm from the datasets review that all data variables are numeric whch is essential for statistical analysis. There is only one categorical variable in the dataset, 'country' which contains a list of all countries used in the study for each model. The country column is not used for statistical analysis as it is a non numerical variable. The variable could be used only when creating visual charts. 


```python
# Checking number of countries or obervations in each dataset

country_count_model_1 = len(model1['country'].unique())
country_count_model_2 = len(model2['country'].unique())
print("Number of countries for Model 1:", country_count_model_1)
print("Number of countries for Model 2:", country_count_model_2)
```

    Number of countries for Model 1: 40
    Number of countries for Model 2: 35
    

The dataset contain 40 countries and 35 countries for model 1 and 2 respectively. The countries are also referred to as the observations. Therefore, we have 40 observations for model 1 and 35 observations for model 2. 

## 5.1. Regression

### Regression Analysis for Model 1


```python
# Indicating and separating the dependent and independent variables from the model 1 dataframe

#Dependent variable definition
dep_var_model1 = model1['mobile_money_account']

#Independent variable definition
indep_var_model1 = model1[['internet_access', 'mobile_utility_payment', 
                           'digital_payment', 'mobile_phone_ownership', 
                           'mobile_bill_payment']]
```


```python
# Adding a constant term to the independent variables for the intercept
indep_var_model1 = sm.add_constant(indep_var_model1)

# Fitting the OLS regression model
ols_model1 = sm.OLS(dep_var_model1, indep_var_model1).fit()

# Printing model summary
print(ols_model1.summary())
```

                                 OLS Regression Results                             
    ================================================================================
    Dep. Variable:     mobile_money_account   R-squared:                       0.858
    Model:                              OLS   Adj. R-squared:                  0.837
    Method:                   Least Squares   F-statistic:                     41.18
    Date:                  Tue, 23 Apr 2024   Prob (F-statistic):           1.77e-13
    Time:                          16:59:14   Log-Likelihood:                -135.17
    No. Observations:                    40   AIC:                             282.3
    Df Residuals:                        34   BIC:                             292.5
    Df Model:                             5                                         
    Covariance Type:              nonrobust                                         
    ==========================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    const                     -7.4068      7.749     -0.956      0.346     -23.155       8.341
    internet_access           -0.5527      0.125     -4.411      0.000      -0.807      -0.298
    mobile_utility_payment     0.7147      0.415      1.721      0.094      -0.129       1.559
    digital_payment            0.6621      0.119      5.579      0.000       0.421       0.903
    mobile_phone_ownership     0.2944      0.161      1.824      0.077      -0.034       0.622
    mobile_bill_payment        0.0607      0.396      0.153      0.879      -0.744       0.865
    ==============================================================================
    Omnibus:                        3.825   Durbin-Watson:                   2.053
    Prob(Omnibus):                  0.148   Jarque-Bera (JB):                2.562
    Skew:                          -0.484   Prob(JB):                        0.278
    Kurtosis:                       3.775   Cond. No.                         618.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    


```python
# Creating a Visual Representation of the Regression Analsis for Model 1

# Getting the predicted values from the OLS model
predicted_values = ols_model1.predict()

# Creating a DataFrame with actual and predicted values
results_df = pd.DataFrame({'Actual': dep_var_model1, 'Predicted': predicted_values})

# Creating a scatter plot with a regression line using Seaborn
sns.lmplot(x='Actual', y='Predicted', data=results_df, line_kws={'color': 'red'})

# Setting plot labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Model 1: Actual vs. Predicted Values')

# Show plot
plt.show()
```


    
![png](output_18_0.png)
    


### Regression Analysis for Model 2


```python
# Indicating and separating the dependent and independent variables from the model 2 dataframe

# Dependent variable definition for model 2
dep_var_model2 = model2['account']

# Independent variable definition for model 2
indep_var_model2 = model2[['source_data_capacity', 'spi_data_use', 
                           'spi_data_services', 'spi_data_products', 
                           'spi_data_sources', 'spi_data_infrastructure']]
```


```python
# Adding a constant term to the independent variables for the intercept
indep_var_model2 = sm.add_constant(indep_var_model2)

# Fitting the OLS regression model for model 2
ols_model2 = sm.OLS(dep_var_model2, indep_var_model2).fit()

# Printing model summary for model 2
print(ols_model2.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                account   R-squared:                       0.612
    Model:                            OLS   Adj. R-squared:                  0.529
    Method:                 Least Squares   F-statistic:                     7.353
    Date:                Tue, 23 Apr 2024   Prob (F-statistic):           8.64e-05
    Time:                        16:59:41   Log-Likelihood:                -136.66
    No. Observations:                  35   AIC:                             287.3
    Df Residuals:                      28   BIC:                             298.2
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    ===========================================================================================
                                  coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------------
    const                       6.8971     25.628      0.269      0.790     -45.600      59.394
    source_data_capacity       -0.5816      0.214     -2.720      0.011      -1.020      -0.144
    spi_data_use               -0.2143      0.243     -0.882      0.385      -0.712       0.284
    spi_data_services          -0.7636      0.228     -3.343      0.002      -1.231      -0.296
    spi_data_products           0.7052      0.485      1.455      0.157      -0.288       1.698
    spi_data_sources            1.1690      0.273      4.276      0.000       0.609       1.729
    spi_data_infrastructure     0.8336      0.289      2.880      0.008       0.241       1.427
    ==============================================================================
    Omnibus:                        2.268   Durbin-Watson:                   2.512
    Prob(Omnibus):                  0.322   Jarque-Bera (JB):                1.155
    Skew:                          -0.237   Prob(JB):                        0.561
    Kurtosis:                       3.753   Cond. No.                     1.72e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.72e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    


```python
# Creating a Visual Representation of the Regression Analsis for Model 2

# Getting the predicted values from the OLS model for model 2
predicted_values_model2 = ols_model2.predict()

# Creating a DataFrame with actual and predicted values for model 2
results_model2 = pd.DataFrame({'Actual': dep_var_model2, 'Predicted': predicted_values_model2})

# Creating a regression plot using Seaborn for model 2
sns.lmplot(x='Actual', y='Predicted', data=results_model2, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})

# Setting plot labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Model 2: Actual vs. Predicted Values')

# Show plot
plt.show()
```


    
![png](output_22_0.png)
    


## 5.2. Pearson Correlation Test

### Correlation Test for Model 1


```python
# creating a correlation matrix for model 1
correlation_matrix_model1 = model1.corr()

# Displaying the correlation matrix
print(correlation_matrix_model1)


# Plotting the heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_model1, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Model 1: Correlation Matrix')
plt.show()
```

                            internet_access  mobile_utility_payment  \
    internet_access                1.000000                0.275408   
    mobile_utility_payment         0.275408                1.000000   
    digital_payment                0.507508                0.736970   
    mobile_phone_ownership         0.856528                0.346597   
    mobile_bill_payment            0.389306                0.940515   
    mobile_money_account           0.092876                0.799250   
    
                            digital_payment  mobile_phone_ownership  \
    internet_access                0.507508                0.856528   
    mobile_utility_payment         0.736970                0.346597   
    digital_payment                1.000000                0.573997   
    mobile_phone_ownership         0.573997                1.000000   
    mobile_bill_payment            0.809292                0.440903   
    mobile_money_account           0.806038                0.279005   
    
                            mobile_bill_payment  mobile_money_account  
    internet_access                    0.389306              0.092876  
    mobile_utility_payment             0.940515              0.799250  
    digital_payment                    0.809292              0.806038  
    mobile_phone_ownership             0.440903              0.279005  
    mobile_bill_payment                1.000000              0.788281  
    mobile_money_account               0.788281              1.000000  
    


    
![png](output_25_1.png)
    


### Correlation Test for Model 2


```python
# creating a correlation matrix for model 2
correlation_matrix_model2 = model2.corr()

# Displaying the correlation matrix
print(correlation_matrix_model1)


# Plotting the heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_model2, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Model 2: Correlation Matrix')
plt.show()
```

                            internet_access  mobile_utility_payment  \
    internet_access                1.000000                0.275408   
    mobile_utility_payment         0.275408                1.000000   
    digital_payment                0.507508                0.736970   
    mobile_phone_ownership         0.856528                0.346597   
    mobile_bill_payment            0.389306                0.940515   
    mobile_money_account           0.092876                0.799250   
    
                            digital_payment  mobile_phone_ownership  \
    internet_access                0.507508                0.856528   
    mobile_utility_payment         0.736970                0.346597   
    digital_payment                1.000000                0.573997   
    mobile_phone_ownership         0.573997                1.000000   
    mobile_bill_payment            0.809292                0.440903   
    mobile_money_account           0.806038                0.279005   
    
                            mobile_bill_payment  mobile_money_account  
    internet_access                    0.389306              0.092876  
    mobile_utility_payment             0.940515              0.799250  
    digital_payment                    0.809292              0.806038  
    mobile_phone_ownership             0.440903              0.279005  
    mobile_bill_payment                1.000000              0.788281  
    mobile_money_account               0.788281              1.000000  
    


    
![png](output_27_1.png)
    


## 5.3. Test for Multicollinearity

### Multicollinearity Test for Model 1


```python
# Creating a DataFrame to store VIF results
vif_data_model1 = pd.DataFrame()
vif_data_model1["Variable"] = indep_var_model1.columns
vif_data_model1["VIF"] = [variance_inflation_factor(indep_var_model1.values, i) for i in range(indep_var_model1.shape[1])]

print("Variance Inflation Factor (VIF) for Model 1:")
print(vif_data_model1)
```

    Variance Inflation Factor (VIF) for Model 1:
                     Variable        VIF
    0                   const  40.483693
    1         internet_access   3.902965
    2  mobile_utility_payment   9.475282
    3         digital_payment   3.495942
    4  mobile_phone_ownership   4.171238
    5     mobile_bill_payment  12.216094
    


```python
# Calculating correlation matrix for independent variables in Model 1
correlation_matrix_model1 = model1[['internet_access', 'mobile_utility_payment', 
                                    'digital_payment', 'mobile_phone_ownership', 
                                    'mobile_bill_payment']].corr()

# Displaying correlation matrix for Model 1
print("Correlation Matrix for Model 1:")
print(correlation_matrix_model1)

# Plotting correlation matrix for Model 1
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_model1, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Model 1: Test for Multicollinearity Matrix')
plt.show()
```

    Correlation Matrix for Model 1:
                            internet_access  mobile_utility_payment  \
    internet_access                1.000000                0.275408   
    mobile_utility_payment         0.275408                1.000000   
    digital_payment                0.507508                0.736970   
    mobile_phone_ownership         0.856528                0.346597   
    mobile_bill_payment            0.389306                0.940515   
    
                            digital_payment  mobile_phone_ownership  \
    internet_access                0.507508                0.856528   
    mobile_utility_payment         0.736970                0.346597   
    digital_payment                1.000000                0.573997   
    mobile_phone_ownership         0.573997                1.000000   
    mobile_bill_payment            0.809292                0.440903   
    
                            mobile_bill_payment  
    internet_access                    0.389306  
    mobile_utility_payment             0.940515  
    digital_payment                    0.809292  
    mobile_phone_ownership             0.440903  
    mobile_bill_payment                1.000000  
    


    
![png](output_31_1.png)
    


### Multicollinearity Test for Model 2


```python
# Creating a DataFrame to store VIF results
vif_data_model2 = pd.DataFrame()
vif_data_model2["Variable"] = indep_var_model2.columns
vif_data_model2["VIF"] = [variance_inflation_factor(indep_var_model2.values, i) for i in range(indep_var_model2.shape[1])]

print("Variance Inflation Factor (VIF) for Model 2:")
print(vif_data_model2)
```

    Variance Inflation Factor (VIF) for Model 2:
                      Variable         VIF
    0                    const  127.526234
    1     source_data_capacity    2.453183
    2             spi_data_use    2.039501
    3        spi_data_services    2.109660
    4        spi_data_products    2.651508
    5         spi_data_sources    2.616190
    6  spi_data_infrastructure    2.604678
    


```python
# Calculatting correlation matrix for independent variables in Model 2
correlation_matrix_model2 = model2[['source_data_capacity', 'spi_data_use', 
                                    'spi_data_services', 'spi_data_products', 
                                    'spi_data_sources', 'spi_data_infrastructure']].corr()

# Displaying correlation matrix for Model 2
print("Correlation Matrix for Model 2:")
print(correlation_matrix_model2)

# Plotting correlation matrix for Model 2
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_model2, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Model 2: Test for Multicollinearity Matrix')
plt.show()
```

    Correlation Matrix for Model 2:
                             source_data_capacity  spi_data_use  \
    source_data_capacity                 1.000000      0.368934   
    spi_data_use                         0.368934      1.000000   
    spi_data_services                    0.199737      0.243448   
    spi_data_products                    0.482582      0.585447   
    spi_data_sources                     0.605898      0.268184   
    spi_data_infrastructure              0.654020      0.539743   
    
                             spi_data_services  spi_data_products  \
    source_data_capacity              0.199737           0.482582   
    spi_data_use                      0.243448           0.585447   
    spi_data_services                 1.000000           0.569445   
    spi_data_products                 0.569445           1.000000   
    spi_data_sources                  0.582020           0.546027   
    spi_data_infrastructure           0.378305           0.439675   
    
                             spi_data_sources  spi_data_infrastructure  
    source_data_capacity             0.605898                 0.654020  
    spi_data_use                     0.268184                 0.539743  
    spi_data_services                0.582020                 0.378305  
    spi_data_products                0.546027                 0.439675  
    spi_data_sources                 1.000000                 0.592323  
    spi_data_infrastructure          0.592323                 1.000000  
    


    
![png](output_34_1.png)
    


## 5.4. Descriptive Statistics

### Descriptive Statistics for Model 1


```python
# Calculating descriptive statistics for model 1
desc_model1 = model1.describe()
desc_model1
```




<div>
<style scoped>
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
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>internet_access</th>
      <th>mobile_utility_payment</th>
      <th>digital_payment</th>
      <th>mobile_phone_ownership</th>
      <th>mobile_bill_payment</th>
      <th>mobile_money_account</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>37.359000</td>
      <td>9.635500</td>
      <td>41.865250</td>
      <td>72.711750</td>
      <td>13.412500</td>
      <td>28.772500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>19.445526</td>
      <td>9.140128</td>
      <td>19.434911</td>
      <td>15.605675</td>
      <td>10.886769</td>
      <td>19.105076</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.220000</td>
      <td>0.000000</td>
      <td>4.810000</td>
      <td>32.160000</td>
      <td>1.450000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>23.115000</td>
      <td>3.142500</td>
      <td>26.740000</td>
      <td>63.490000</td>
      <td>4.937500</td>
      <td>9.782500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>31.055000</td>
      <td>6.225000</td>
      <td>42.985000</td>
      <td>75.665000</td>
      <td>9.855000</td>
      <td>31.865000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>49.785000</td>
      <td>13.865000</td>
      <td>54.182500</td>
      <td>83.212500</td>
      <td>18.960000</td>
      <td>42.477500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>82.910000</td>
      <td>36.060000</td>
      <td>80.810000</td>
      <td>100.000000</td>
      <td>44.910000</td>
      <td>68.660000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Selecting only numeric columns for calculating descriptive statistics
numeric_cols_model1 = ['mobile_money_account','internet_access', 'mobile_utility_payment', 
                       'digital_payment', 'mobile_phone_ownership', 
                       'mobile_bill_payment']

desc_model1 = model1[numeric_cols_model1].describe()

# Adding skewness and kurtosis
desc_model1.loc['skewness'] = model1[numeric_cols_model1].skew()
desc_model1.loc['kurtosis'] = model1[numeric_cols_model1].kurt()

# Adding additional metrics
desc_model1.loc['Standard Error'] = model1[numeric_cols_model1].sem()
desc_model1.loc['Mode'] = model1[numeric_cols_model1].mode().iloc[0]  # Get the mode from the first row
desc_model1.loc['Sample Variance'] = model1[numeric_cols_model1].var()
desc_model1.loc['Range'] = model1[numeric_cols_model1].max() - model1[numeric_cols_model1].min()
desc_model1.loc['Sum'] = model1[numeric_cols_model1].sum()

# Displaying the summary
desc_model1
```




<div>
<style scoped>
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
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mobile_money_account</th>
      <th>internet_access</th>
      <th>mobile_utility_payment</th>
      <th>digital_payment</th>
      <th>mobile_phone_ownership</th>
      <th>mobile_bill_payment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>28.772500</td>
      <td>37.359000</td>
      <td>9.635500</td>
      <td>41.865250</td>
      <td>72.711750</td>
      <td>13.412500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>19.105076</td>
      <td>19.445526</td>
      <td>9.140128</td>
      <td>19.434911</td>
      <td>15.605675</td>
      <td>10.886769</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>5.220000</td>
      <td>0.000000</td>
      <td>4.810000</td>
      <td>32.160000</td>
      <td>1.450000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.782500</td>
      <td>23.115000</td>
      <td>3.142500</td>
      <td>26.740000</td>
      <td>63.490000</td>
      <td>4.937500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>31.865000</td>
      <td>31.055000</td>
      <td>6.225000</td>
      <td>42.985000</td>
      <td>75.665000</td>
      <td>9.855000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>42.477500</td>
      <td>49.785000</td>
      <td>13.865000</td>
      <td>54.182500</td>
      <td>83.212500</td>
      <td>18.960000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>68.660000</td>
      <td>82.910000</td>
      <td>36.060000</td>
      <td>80.810000</td>
      <td>100.000000</td>
      <td>44.910000</td>
    </tr>
    <tr>
      <th>skewness</th>
      <td>0.092837</td>
      <td>0.692206</td>
      <td>1.338398</td>
      <td>0.238878</td>
      <td>-0.677023</td>
      <td>1.395810</td>
    </tr>
    <tr>
      <th>kurtosis</th>
      <td>-1.058531</td>
      <td>-0.184415</td>
      <td>1.173069</td>
      <td>-0.652349</td>
      <td>0.111974</td>
      <td>1.681350</td>
    </tr>
    <tr>
      <th>Standard Error</th>
      <td>3.020778</td>
      <td>3.074608</td>
      <td>1.445181</td>
      <td>3.072929</td>
      <td>2.467474</td>
      <td>1.721349</td>
    </tr>
    <tr>
      <th>Mode</th>
      <td>29.380000</td>
      <td>5.220000</td>
      <td>13.610000</td>
      <td>33.740000</td>
      <td>82.760000</td>
      <td>1.450000</td>
    </tr>
    <tr>
      <th>Sample Variance</th>
      <td>365.003927</td>
      <td>378.128496</td>
      <td>83.541943</td>
      <td>377.715749</td>
      <td>243.537097</td>
      <td>118.521737</td>
    </tr>
    <tr>
      <th>Range</th>
      <td>68.660000</td>
      <td>77.690000</td>
      <td>36.060000</td>
      <td>76.000000</td>
      <td>67.840000</td>
      <td>43.460000</td>
    </tr>
    <tr>
      <th>Sum</th>
      <td>1150.900000</td>
      <td>1494.360000</td>
      <td>385.420000</td>
      <td>1674.610000</td>
      <td>2908.470000</td>
      <td>536.500000</td>
    </tr>
  </tbody>
</table>
</div>



### Descriptive Statistics for Model 2


```python
# Calculating descriptive statistics for model 2
desc_model2 = model2.describe()

# Displaying the summary
desc_model2
```




<div>
<style scoped>
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
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source_data_capacity</th>
      <th>spi_data_use</th>
      <th>spi_data_services</th>
      <th>spi_data_products</th>
      <th>spi_data_sources</th>
      <th>spi_data_infrastructure</th>
      <th>account</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>35.000000</td>
      <td>35.000000</td>
      <td>35.000000</td>
      <td>35.000000</td>
      <td>35.000000</td>
      <td>35.000000</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>55.142857</td>
      <td>76.851429</td>
      <td>61.064857</td>
      <td>78.389429</td>
      <td>32.927143</td>
      <td>51.857143</td>
      <td>48.731429</td>
    </tr>
    <tr>
      <th>std</th>
      <td>16.868249</td>
      <td>13.529605</td>
      <td>14.641475</td>
      <td>7.736023</td>
      <td>13.621253</td>
      <td>12.838689</td>
      <td>19.553514</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.000000</td>
      <td>40.000000</td>
      <td>20.600000</td>
      <td>53.920000</td>
      <td>9.040000</td>
      <td>25.000000</td>
      <td>5.830000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>45.000000</td>
      <td>70.000000</td>
      <td>60.300000</td>
      <td>76.540000</td>
      <td>25.590000</td>
      <td>42.500000</td>
      <td>34.560000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>60.000000</td>
      <td>80.000000</td>
      <td>63.870000</td>
      <td>78.960000</td>
      <td>29.180000</td>
      <td>50.000000</td>
      <td>49.490000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>65.000000</td>
      <td>90.000000</td>
      <td>67.200000</td>
      <td>82.445000</td>
      <td>38.620000</td>
      <td>60.000000</td>
      <td>61.690000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.000000</td>
      <td>100.000000</td>
      <td>86.670000</td>
      <td>89.390000</td>
      <td>70.280000</td>
      <td>80.000000</td>
      <td>90.530000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Selecting only numeric columns for calculating descriptive statistics
numeric_cols_model2 = ['account','source_data_capacity', 'spi_data_use', 
                'spi_data_services', 'spi_data_products', 
                'spi_data_sources', 'spi_data_infrastructure']

desc_model2 = model2[numeric_cols_model2].describe()

# Adding skewness and kurtosis
desc_model2.loc['skewness'] = model2[numeric_cols_model2].skew()
desc_model2.loc['kurtosis'] = model2[numeric_cols_model2].kurt()

# Adding additional metrics
desc_model2.loc['Standard Error'] = model2[numeric_cols_model2].sem()
desc_model2.loc['Mode'] = model2[numeric_cols_model2].mode().iloc[0]  # Get the mode from the first row
desc_model2.loc['Sample Variance'] = model2[numeric_cols_model2].var()
desc_model2.loc['Range'] = model2[numeric_cols_model2].max() - model2[numeric_cols_model2].min()
desc_model2.loc['Sum'] = model2[numeric_cols_model2].sum()

# Displaying the summary
desc_model2
```




<div>
<style scoped>
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
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>account</th>
      <th>source_data_capacity</th>
      <th>spi_data_use</th>
      <th>spi_data_services</th>
      <th>spi_data_products</th>
      <th>spi_data_sources</th>
      <th>spi_data_infrastructure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>35.000000</td>
      <td>35.000000</td>
      <td>35.000000</td>
      <td>35.000000</td>
      <td>35.000000</td>
      <td>35.000000</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>48.731429</td>
      <td>55.142857</td>
      <td>76.851429</td>
      <td>61.064857</td>
      <td>78.389429</td>
      <td>32.927143</td>
      <td>51.857143</td>
    </tr>
    <tr>
      <th>std</th>
      <td>19.553514</td>
      <td>16.868249</td>
      <td>13.529605</td>
      <td>14.641475</td>
      <td>7.736023</td>
      <td>13.621253</td>
      <td>12.838689</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.830000</td>
      <td>20.000000</td>
      <td>40.000000</td>
      <td>20.600000</td>
      <td>53.920000</td>
      <td>9.040000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>34.560000</td>
      <td>45.000000</td>
      <td>70.000000</td>
      <td>60.300000</td>
      <td>76.540000</td>
      <td>25.590000</td>
      <td>42.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>49.490000</td>
      <td>60.000000</td>
      <td>80.000000</td>
      <td>63.870000</td>
      <td>78.960000</td>
      <td>29.180000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>61.690000</td>
      <td>65.000000</td>
      <td>90.000000</td>
      <td>67.200000</td>
      <td>82.445000</td>
      <td>38.620000</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.530000</td>
      <td>80.000000</td>
      <td>100.000000</td>
      <td>86.670000</td>
      <td>89.390000</td>
      <td>70.280000</td>
      <td>80.000000</td>
    </tr>
    <tr>
      <th>skewness</th>
      <td>-0.048651</td>
      <td>-0.377096</td>
      <td>-0.616829</td>
      <td>-1.168967</td>
      <td>-1.191027</td>
      <td>0.971202</td>
      <td>0.102107</td>
    </tr>
    <tr>
      <th>kurtosis</th>
      <td>-0.076453</td>
      <td>-0.434600</td>
      <td>0.417437</td>
      <td>1.996276</td>
      <td>1.960332</td>
      <td>1.244482</td>
      <td>-0.361374</td>
    </tr>
    <tr>
      <th>Standard Error</th>
      <td>3.305147</td>
      <td>2.851254</td>
      <td>2.286921</td>
      <td>2.474861</td>
      <td>1.307627</td>
      <td>2.302412</td>
      <td>2.170135</td>
    </tr>
    <tr>
      <th>Mode</th>
      <td>5.830000</td>
      <td>60.000000</td>
      <td>80.000000</td>
      <td>57.030000</td>
      <td>53.920000</td>
      <td>9.040000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>Sample Variance</th>
      <td>382.339924</td>
      <td>284.537815</td>
      <td>183.050218</td>
      <td>214.372796</td>
      <td>59.846053</td>
      <td>185.538533</td>
      <td>164.831933</td>
    </tr>
    <tr>
      <th>Range</th>
      <td>84.700000</td>
      <td>60.000000</td>
      <td>60.000000</td>
      <td>66.070000</td>
      <td>35.470000</td>
      <td>61.240000</td>
      <td>55.000000</td>
    </tr>
    <tr>
      <th>Sum</th>
      <td>1705.600000</td>
      <td>1930.000000</td>
      <td>2689.800000</td>
      <td>2137.270000</td>
      <td>2743.630000</td>
      <td>1152.450000</td>
      <td>1815.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Plotting Skewness for Model 1 and 2


```python
# Skewness data for Model 1
skewness_model1 = [0.09, 0.69, 1.34, 0.24, -0.68, 1.40]

# Variable names for Model 2
variables_model1 = ['MMA', 'IA', 'MUP', 'DP', 'MPO', 'MBP']

# Plotting skewness for Model 2
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=skewness_model1, y=variables_model1, palette="viridis")
plt.title('Skewness for Model 1 Variables')
plt.xlabel('Skewness')
plt.ylabel('Variables')

# Adding data points to the bars
for i in range(len(skewness_model1)):
    ax.text(skewness_model1[i], i, f'{skewness_model1[i]:.2f}', ha='left', va='center', color='black')
    
plt.show()
```


    
![png](output_43_0.png)
    



```python
# Skewness data for Model 2
skewness_model2 = [-0.05, -0.38, -0.62, -1.17, -1.19, 0.97, 0.10]

# Variable names for Model 2
variables_model2 = ['ACC', 'SDC', 'DUS', 'DSS', 'DPS', 'DSC', 'DIS']

# Plotting skewness for Model 2
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=skewness_model2, y=variables_model2, palette="viridis")
plt.title('Skewness for Model 2 Variables')
plt.xlabel('Skewness')
plt.ylabel('Variables')

# Adding data points to the bars
for i in range(len(skewness_model2)):
    ax.text(skewness_model2[i], i, f'{skewness_model2[i]:.2f}', ha='left', va='center', color='black')
    
plt.show()
```


    
![png](output_44_0.png)
    


## 5.5. Hypothesis Testing

### Hypothesis Testing for Model 1


```python
# Line of Best fit for Model 1
ols_model1

# Overall Model Significance Test (F-test) for Model 1
f_statistic, f_p_value = ols_model1.fvalue, ols_model1.f_pvalue
print("Model 1 F-statistic:", f_statistic)
print("Model 1 p-value (F-test):", f_p_value)

# Individual Coefficient Significance Test (t-test) for Model 1
t_test_results = ols_model1.t_test(np.eye(len(ols_model1.params)))

print("Individual Coefficient p-values (t-test) for Model 1:")
for idx, variable in enumerate(indep_var_model1.columns):
    print(f"{variable}: {t_test_results.pvalue[idx]}")
```

    Model 1 F-statistic: 41.184773278043544
    Model 1 p-value (F-test): 1.7719819035065024e-13
    Individual Coefficient p-values (t-test) for Model 1:
    const: 0.3459100134441736
    internet_access: 9.831638939501509e-05
    mobile_utility_payment: 0.09440462482524704
    digital_payment: 3.032239454019748e-06
    mobile_phone_ownership: 0.07693681973223013
    mobile_bill_payment: 0.8790651297196372
    

### Hypothesis Testing for Model 2


```python
# Line of best fit for Model 2
ols_model2

# Overall Model Significance Test (F-test) for Model 2
f_statistic, f_p_value = ols_model2.fvalue, ols_model2.f_pvalue
print("Model 2 F-statistic:", f_statistic)
print("Model 2 p-value (F-test):", f_p_value)

# Individual Coefficient Significance Test (t-test) for Model 2
t_test_results = ols_model2.t_test(np.eye(len(ols_model2.params)))
print("Individual Coefficient p-values (t-test) for Model 2:")
for idx, variable in enumerate(indep_var_model2.columns):
    print(f"{variable}: {t_test_results.pvalue[idx]}")
```

    Model 2 F-statistic: 7.352573148811251
    Model 2 p-value (F-test): 8.644605466870827e-05
    Individual Coefficient p-values (t-test) for Model 2:
    const: 0.7898080501069904
    source_data_capacity: 0.011083716383894056
    spi_data_use: 0.38538856115979137
    spi_data_services: 0.0023663090448723797
    spi_data_products: 0.15678092693555912
    spi_data_sources: 0.0001999381497366182
    spi_data_infrastructure: 0.0075410358992607385
    

# 6. Phase 5: Share


```python
# Exporting regression result as an Excel File
# Convert the regression results summary to a dataframe
regression_results_model1 = ols_model1.summary()
regression_results_df_model1 = pd.read_html(regression_results_model1.tables[1].as_html(), header=0, index_col=0)[0]

# Export the dataframe to an Excel file
regression_results_df_model1.to_excel("regression_results_model1.xlsx")

print("Regression results for Model 1.xlsx")
```

    Regression results for Model 1.xlsx
    


```python
# Exporting descriptive stats result as an excel file for Model 1
desc_model1.to_excel('desc_model1.xlsx')

print('Descriptive statistics for Model 1.xlsx')
```

    Descriptive statistics for Model 1.xlsx
    


```python
# Exporting descriptive stats result as an excel file for Model 2
desc_model2.to_excel('desc_model2.xlsx')

print('Descriptive statistics for Model 2.xlsx')
```

    Descriptive statistics for Model 2.xlsx
    


```python
predicted_values_model2
```




    array([36.25275297, 67.71857471, 52.46712807, 47.98902335, 18.8266856 ,
           28.58058205, 44.9136686 , 53.01701255, 67.08734359, 50.64486208,
           56.00199781, 44.26275099, 62.90811648, 39.46190352, 59.06135446,
           61.11429563, 45.06825223, 38.74550916, 53.30689915, 36.97488527,
           33.13351753, 70.41003994, 44.86373269, 43.75912241, 46.66711801,
           40.7967662 , 62.99003982, 34.50329914, 99.3950081 , 19.99720639,
           41.06566196, 50.30056932, 51.74385254, 41.88368771, 59.68677996])



# 7. Phase 6: Conclusion (Act)


```python

```


```python

```


```python

```
