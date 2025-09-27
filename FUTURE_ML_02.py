#!/usr/bin/env python
# coding: utf-8

# In[146]:


import sys
print(sys.version)
import numpy as np
import scipy as sc 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from  sklearn.preprocessing import LabelEncoder, StandardScaler
import imbalanced_learn as imblearn
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble  import RandomForestClassifier
from XGBoost  import XGBclassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle


# In[147]:


file_path = r"C:\Users\koskr\Desktop\future_inturn\task_2\archive\Churn_Modelling.csv"


# In[148]:


df = pd.read_csv(file_path)
print(df.head())


# In[176]:


file_path1 = r"C:\Users\koskr\Desktop\future_inturn\task_2\archive\Churn_Modelling.csv"


# In[177]:


df = pd.read_csv(file_path1)
print(df.head())


# In[178]:


file_path2 = r"C:\Users\koskr\Desktop\future_inturn\task_2\spotify-user-behavior-dataset\Spotify_data.xlsx"


# In[182]:





# In[149]:


print(f"Number of rows/datasets in dataframe : {len(df)}")


# In[150]:


print(df.columns)


# In[151]:


df.shape


# In[152]:


df.columns = df.columns.str.strip()


# In[ ]:





# In[ ]:





# In[ ]:





# In[153]:


df.head()


# In[154]:


pd.set_option("display.max_columns", 0)
df.head(1)


# In[155]:


df.info()


# In[156]:


for each_dtype, each_column in zip(df.dtypes, df.columns):
    if (each_dtype == 'object'):
        print(f"{each_column} has total of {len(df[each_column].unique())} types of category")


# In[157]:


df.head()


# In[186]:


df.replace(" ", np.nan, inplace=True)


# In[187]:


df.dropna(inplace=True)


# In[188]:


df.drop('CustomerId', axis=1, inplace=True)
df.head(2)


# In[192]:


print(df.columns)


# In[193]:


df.columns = df.columns.str.strip()  # Removes leading/trailing spaces


# In[194]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


# In[190]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])


# In[191]:


print(df.columns)  # Check all column names in the DataFrame


# In[160]:


df.columns = df.columns.str.strip()


# In[ ]:





# In[161]:


print(df.head())  # Check the first few rows of the DataFrame


# In[162]:


df.head(2)


# In[163]:


print(df.get('TotalCharges', 'Column not found'))


# In[164]:


print(df.columns)


# In[165]:


columns_list = df.columns.tolist()
print(columns_list)


# In[166]:


print(df["Gender"].unique())


# In[167]:


df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [25, 30]})
print(df.columns)

if 'Gender' in df.columns:
    print(df['Gender'])
else:
    print("Column not exist.")


# In[168]:


numerical_features_list = ["tenure", "MonthlyCharges", "TotalCharges"]


# In[169]:


for col in df.columns:
    if col not in numerical_features_list:
        print(col,df[col].unique())
        print("  _  ",50)


# In[170]:


print(df.isnull().sum())


# In[171]:


df.info()


# In[175]:


df = df.astype({
    'TotalCharges' : 'float',
})


# In[174]:


df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [25, 30]})
print(df.columns)

if 'TotalCharges' in df.columns:
    print(df['TotalCharges'])
else:
    print("Column not exist.")


# In[173]:


df[df['TotalCharges']==" "]


# In[79]:


df["TotalCharges"].replace(' ', np.nan, inplace=True)


# In[76]:


for each in df.columns:
    count = 0
    for each_row in df[each]:
        if each_row == ' ':
            count += 1
    print(f"Number of empty values in column {each} is {count}.")


# In[78]:


df.loc[df.TotalCharges == ' ' ,'TotalCharges']


# In[145]:


print(df['TotalCharges'].unique())


# In[ ]:





# In[ ]:





# In[ ]:





# In[204]:


# Label Encode binary categories, one-hot encode the rest
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() == 2:
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col])


# In[205]:


X = df.drop("Gender", axis=1)  # Features
y = df["Gender"]               # Target


# In[206]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[207]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[208]:


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("Logistic Regression:")
print(classification_report(y_test, y_pred_lr))


# In[213]:


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest:")
print(classification_report(y_test, y_pred_rf))


# In[214]:


xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost:")
print(classification_report(y_test, y_pred_xgb))


# In[ ]:




