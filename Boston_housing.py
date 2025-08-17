
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



col_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS","RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]



df=pd.read_csv(r"C:\Users\aksha\OneDrive\Desktop\datasetss\housing.csv",sep=r'\s+',names=col_names)
df


# | Column   | Meaning  |
# |----------|-------------------------|
# | CRIM     | Crime rate per person in the town |
# | ZN       | % of land for large houses |
# | INDUS    | % of land for businesses (not houses) |
# | CHAS     | Near Charles River? (1 = yes, 0 = no) |
# | NOX      | Air pollution level (nitric oxide) |
# | RM       | Average number of rooms per house |
# | AGE      | % of houses built before 1940 |
# | DIS      | Distance from job centers |
# | RAD      | Accessibility to highways |
# | TAX      | Property tax rate |
# | PTRATIO  | Students per teacher in schools |
# | B        | Demographic formula from census |
# | LSTAT    | % of low-income residents |
# | MEDV     | Median house value in $1000s |
# 



df.isnull().sum()



df.duplicated().sum()


# In[123]:


df.info()


# In[124]:


df.describe().round(2)


# In[125]:


df.shape


# In[126]:


def draw_boxplots(df):
    plt.figure(figsize=(15,15))
    plt.suptitle("Boxplot Distribution", fontsize=20, fontweight='bold')

    numerical_features=df.select_dtypes(include=np.number).columns

    for i,feature in enumerate(numerical_features):
        plt.subplot(5,3,i+1)
        sns.boxplot(data=df[feature])
        plt.title(f"Boxplot of {feature}")

    plt.tight_layout(rect=[0,0,0,0.95])
    plt.show()


# In[127]:


draw_boxplots(df)


# In[128]:


df.skew()


# In[129]:


correlation_matrix=df.corr()
plt.figure(figsize=(15,8))
sns.heatmap(data=correlation_matrix,annot=True,xticklabels=True,yticklabels=True,cmap="coolwarm")


# In[130]:


correlation_matrix[abs(correlation_matrix)>0.7].round(2)


# In[ ]:





# In[131]:


#ignore self correltion
no_self_col=correlation_matrix.copy()
np.fill_diagonal(no_self_col.values,0)

#keep variables with atleast one value >0.7
filtered_columns=no_self_col.columns[(abs(no_self_col)>0.7).any(axis=0)]

#creating filetered correlation_matrix
filtered_corr_matrix=correlation_matrix.loc[filtered_columns,filtered_columns]

plt.figure(figsize=(8, 6))
sns.heatmap(filtered_corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("High Multicollinearity (|corr| > 0.7)")
plt.show()


# In[132]:


outlier_treat_col=["CRIM", "ZN", "RM","B","TAX","RAD","PTRATIO","LSTAT"]

def outlier_treatment(df,features):
    for col in features:
        Q1=df[col].quantile(0.25)
        Q3=df[col].quantile(0.75)

        IQR=Q3-Q1

        lower_bound=Q1-1.5*IQR
        upper_bound=Q3+1.5*IQR

        df[col]=np.where(df[col]<lower_bound,lower_bound,df[col])
        df[col]=np.where(df[col]>upper_bound,upper_bound,df[col])

    return df



# In[133]:


df=outlier_treatment(df, outlier_treat_col)
df.describe()


# In[134]:


def draw_boxplots(df):
    plt.figure(figsize=(15,15))
    plt.suptitle("Boxplot Distribution", fontsize=20, fontweight='bold')

    numerical_features=df.select_dtypes(include=np.number).columns

    for i,feature in enumerate(numerical_features):
        plt.subplot(5,3,i+1)
        sns.boxplot(data=df[feature])
        plt.title(f"Boxplot of {feature}")

    plt.tight_layout(rect=[0,0,0,0.95])
    plt.show()




# In[135]:


draw_boxplots(df)


# Split Features into Independent and Dependent Variable

# In[136]:


x=df.drop(['MEDV'],axis=1)
y=df['MEDV']
print(f"x_shape = {x.shape}")
print(f"y_shape = {y.shape}")


# Feature Scaling

# In[137]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaler=pd.DataFrame(scaler.fit_transform(x),columns=x.columns)
x_scaler


# In[138]:


x_scaler.describe().round(2)


# VARIANCE INFLATION FACTOR

# In[ ]:





# In[139]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def variance_inflation(x_scaler):
    variable=x_scaler
    vif=pd.DataFrame()
    vif["variance_inflation_factor"]=[variance_inflation_factor(variable,i) for i in range(variable.shape[1])]
    vif["columns"]=x_scaler.columns
    return vif


# In[140]:


variance_inflation(x_scaler).round(2)


# Variance Inflation Factor of RAD > 10   its better to remove it

# In[141]:


x_scaler=x_scaler.drop(["RAD"],axis=1)


# In[142]:


variance_inflation(x_scaler).round(2)


# In[143]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

print(f"x_train = {x_train.shape},y_train = {y_train.shape},\nx_test = {x_test.shape}, y_test = {y_test.shape}, ")


# In[144]:


from statsmodels.regression.linear_model import OLS
import statsmodels.regression.linear_model as smf


# In[145]:


import statsmodels.api as sm
x_train_ols= sm.add_constant(x_train) 
x_train_ols


# In[146]:


regression = smf.OLS(endog=y_train, exog= x_train_ols).fit()
regression.summary()


# In[147]:


from sklearn.linear_model import LinearRegression
reg_model=LinearRegression()
reg_model.fit(x_train,y_train)


# In[148]:


print("co-efficient :", reg_model.coef_)


# In[149]:


print("intercept :", reg_model.intercept_)


# In[150]:


y_pred=reg_model.predict(x_test)
y_pred


# In[ ]:





# In[151]:


from sklearn.metrics import r2_score
print(f"Accuracy = {r2_score(y_test,y_pred)}")


# In[152]:


print(x_train.shape,y_train.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[153]:


residuals = y_pred - y_test  
plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals, color='dodgerblue', alpha=0.6, label='Residualss')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)  # reference line at 0
plt.xlabel('Predicted Values', fontsize=15)
plt.ylabel('Residuals', fontsize=15)
plt.title("Residual Plot", fontsize=18)
plt.legend(fontsize=12)
plt.show()



# In[154]:


sm.qqplot(residuals, line='45')
plt.show()


# In[155]:


from sklearn.ensemble import RandomForestRegressor
randomforest_model = RandomForestRegressor()
randomforest_model.fit(x_train, y_train)


# In[156]:


y_pred_train_rf = randomforest_model.predict(x_train)
y_pred_test_rf= randomforest_model.predict(x_test)


# In[157]:


print("training accuracy", r2_score(y_train, y_pred_train_rf))
print()
print("test accuracy", r2_score(y_test, y_pred_test_rf))


# In[ ]:





# In[ ]:





# In[158]:


y_pred_train_rf = randomforest_model.predict(x_train)
y_pred_test_rf= randomforest_model.predict(x_test)


# In[159]:


print("training accuracy", r2_score(y_train, y_pred_train_rf))
print()
print("test accuracy", r2_score(y_test, y_pred_test_rf))


# In[ ]:





# In[ ]:





# In[160]:


from sklearn.metrics import mean_squared_error
import numpy as np

# MSE
mse = mean_squared_error(y_test, y_pred_test_rf)

# RMSE (manual square root)
rmse = np.sqrt(mse)

print("Train R²:", r2_score(y_train, y_pred_train_rf))
print("Test R²:", r2_score(y_test, y_pred_test_rf))
print("Test RMSE:", rmse)


# In[161]:


importances = randomforest_model.feature_importances_
features = df.drop("MEDV", axis=1).columns  # assuming 'medv' is target

plt.figure(figsize=(10,6))
plt.barh(features, importances, color="dodgerblue")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importances")
plt.show()


# In[162]:


# Save randomforest_model
import pickle
with open("boston_randomforest_model.pkl", "wb") as f:
    pickle.dump(randomforest_model, f)

# Prediction function
def predict_price(features):
    """
    Predict house price based on input features.
    features: list or array with length = 13 (Boston dataset features)
    """
    features = np.array(features).reshape(1, -1)
    with open("boston_randomforest_model.pkl", "rb") as f:
        randomforest_model = pickle.load(f)
    return randomforest_model.predict(features)[0]


# In[ ]:




