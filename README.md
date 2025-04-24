## EXNO-3-DS
## REG NO: 212223040241
# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/b408ef2e-0243-4b74-8d3c-6d717b716d43)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/ff439eb0-c0b3-41ed-b636-2a61b15aa3d8)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/0f95ca90-ab59-4684-8a4a-48c071fee9e8)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/7d943205-0786-4c85-8792-b42f236c41d7)
```
from sklearn.preprocessing import OneHotEncoder
df
```
![image](https://github.com/user-attachments/assets/26c8af3b-d4f1-439a-b408-f8f5061b07d6)
```
ohe=OneHotEncoder(sparse_output=False)
df=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df[['nom_0']]))
enc
```
![image](https://github.com/user-attachments/assets/f908a4a8-898d-4a94-86d4-b241d65bd938)
```
df2=pd.concat([df,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/67223f50-1a2e-4892-93d2-e416f9cf7e0c)
```
from category_encoders import BinaryEncoder
df=pd.read_csv(r"C:\Users\admin\Downloads\data (1).csv")
df
```
![image](https://github.com/user-attachments/assets/9689f029-b221-46ee-b497-cdeea5450eb4)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/d52e1ec1-fa4d-4866-b4ad-159ffc3f776a)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/e64bb4ba-0a92-4bf5-b480-9f47dbbee0ff)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv(r"C:\Users\admin\Downloads\Data_to_Transform.csv")
df.skew()
```
![image](https://github.com/user-attachments/assets/7e6f798c-2b8e-453a-8564-0bdfbcbf7b19)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/85f15875-16a8-4df0-baf5-23a7669e125e)
```
np.reciprocal(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/e65ce239-7538-43b6-b0aa-15408cd57562)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/4005eb48-5864-479e-be6d-5a3c93eacb39)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/2c3b56b8-d5ad-436a-8b7a-77d9b657f572)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/5d01e9d5-cb0c-496e-993e-0a990115a4be)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/651e0e4a-4f28-4dfb-a9e8-969ae83c2c25)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df['Moderate Negative Skew'])
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/f622813d-f34f-47e2-a08b-506ac7f1fb7f)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/3f534aff-8d40-4418-85aa-80161f15b385)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/0f38e86c-5846-43cb-a8b3-fa243b4c23c8)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/ef881915-c708-4398-bb62-350645e1147a)





















# RESULT:
      Thus the code for Data Transformation is executed successfully.

       
