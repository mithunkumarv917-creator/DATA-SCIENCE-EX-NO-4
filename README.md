# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
```
Import required libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

Feature scaling using BMI.csv

Load dataset

df = pd.read_csv('bmi.csv')
print("Original Dataset:")
print(df.head())

Handle missing values

df = df.dropna()

1 - Standard Scaling

df_std = df.copy()
scaler_std = StandardScaler()
df_std[['Height', 'Weight']] = scaler_std.fit_transform(df_std[['Height', 'Weight']])
print("\nStandard Scaled Data:")
print(df_std.head())

2 - Min-Max Scaling

df_minmax = df.copy()
scaler_minmax = MinMaxScaler()
df_minmax[['Height', 'Weight']] = scaler_minmax.fit_transform(df_minmax[['Height', 'Weight']])
print("\nMin-Max Scaled Data:")
print(df_minmax.head())

3 - MaxAbs Scaling

df_maxabs = df.copy()
scaler_maxabs = MaxAbsScaler()
df_maxabs[['Height', 'Weight']] = scaler_maxabs.fit_transform(df_maxabs[['Height', 'Weight']])
print("\nMaxAbs Scaled Data:")
print(df_maxabs.head())

4 - Robust Scaling

df_robust = df.copy()
scaler_robust = RobustScaler()
df_robust[['Height', 'Weight']] = scaler_robust.fit_transform(df_robust[['Height', 'Weight']])
print("\nRobust Scaled Data:")
print(df_robust.head())

Save scaled datasets

#df_std.to_csv("BMI_StandardScaled.csv", index=False)
#df_minmax.to_csv("BMI_MinMaxScaled.csv", index=False)
#df_maxabs.to_csv("BMI_MaxAbsScaled.csv", index=False)
#df_robust.to_csv("BMI_RobustScaled.csv", index=False)
print("\nFeature Scaling Completed Successfully.")

```
<img width="357" height="311" alt="554266881-c7dfbbcb-f09a-4a57-a6b8-e5fff4c661fc" src="https://github.com/user-attachments/assets/06f6e2c8-39ae-4570-98ef-2ad1c1359ca9" />
<img width="886" height="410" alt="554267041-7c494b5f-60f6-43ae-896d-c340da7e4dad" src="https://github.com/user-attachments/assets/bf24e38e-dc7d-4f69-8e76-7d9c1c24d2f2" />
<img width="975" height="397" alt="554267143-ba573a33-7e2c-4b6c-bc1c-bfe19b5f69de" src="https://github.com/user-attachments/assets/2ec82805-5f15-4af2-abe1-193317e7c0eb" />
<img width="964" height="375" alt="554267205-f3c1ea14-1071-479d-9f99-84735bb25674" src="https://github.com/user-attachments/assets/1114683e-3fd3-43ee-b4b5-f2c1208285d2" />
<img width="962" height="379" alt="554267270-bc854394-a05a-4151-9951-95c6d6a66114" src="https://github.com/user-attachments/assets/69f2d390-3bfe-428e-a2aa-483bbb3e33e3" />
<img width="585" height="238" alt="554267366-b04736a7-85ac-4485-9692-82dc3ed331f7" src="https://github.com/user-attachments/assets/083eb6a8-e873-42ad-ab4b-9cff7e922426" />

```
Import required libraries

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

Load Dataset

df = pd.read_csv("income.csv")
print("Dataset Preview:")
print(df.head())

Encode Categorical Variables

categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation',
'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category').apply(lambda x: x.cat.codes)

Encode Target Variable if needed

if df['SalStat'].dtype == 'object':
    df['SalStat'] = df['SalStat'].astype('category').cat.codes

Separate Features and Target

X = df.drop(columns=['SalStat'])
y = df['SalStat']

Scale Data for Chi-Square (Non-negative required)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

Filter Method: Chi-Square

selector_chi2 = SelectKBest(score_func=chi2, k=6)
selector_chi2.fit(X_scaled, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("\nChi-Square Selected:", list(selected_features_chi2))

Filter Method: ANOVA

selector_anova = SelectKBest(score_func=f_classif, k=5)
selector_anova.fit(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nANOVA Selected:", list(selected_features_anova))

Wrapper Method: RFE

logreg = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=logreg, n_features_to_select=6)
rfe.fit(X, y)
selected_features_rfe = X.columns[rfe.support_]
print("\nRFE Selected:", list(selected_features_rfe))

Embedded Method: SelectFromModel

rf = RandomForestClassifier(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
rf.fit(X_train, y_train)
selector_embedded = SelectFromModel(rf, threshold="mean")
selector_embedded.fit(X_train, y_train)

selected_features_embedded = X.columns[selector_embedded.get_support()]
print("\nEmbedded Method Selected:", list(selected_features_embedded))

Accuracy using Embedded Features

X_train_sel = selector_embedded.transform(X_train)
X_test_sel = selector_embedded.transform(X_test)
rf.fit(X_train_sel, y_train)
y_pred = rf.predict(X_test_sel)
print("\nModel Accuracy (Embedded Method):", accuracy_score(y_test, y_pred))
```

<img width="808" height="627" alt="554269288-62a1280b-3bee-448d-97ca-ec507d54129a" src="https://github.com/user-attachments/assets/dcb7b085-b541-4d27-94a4-42a402fba5c5" />
<img width="1035" height="224" alt="554269455-177453ae-61e7-4ca9-aa06-69ed980d3f21" src="https://github.com/user-attachments/assets/37a88e1f-7326-41f2-8b8f-aef304893552" />
<img width="1035" height="224" alt="554269586-063c22f2-8c67-4fd1-a6ec-2a70a98b006a" src="https://github.com/user-attachments/assets/2c6d9266-2e3d-4c0d-b052-d395c7b6c5d1" />
<img width="1035" height="224" alt="554269722-bea9adf2-7c5c-43fd-b871-f8e8a1f4f09a" src="https://github.com/user-attachments/assets/14381880-db90-47ad-90e7-e4f33cbaae66" />
<img width="1035" height="224" alt="554269867-778a5a1c-89df-4728-b2e4-47998a57d9f2" src="https://github.com/user-attachments/assets/06daf2e3-5068-4cb4-9c19-fffdb1f32ecc" />
<img width="1035" height="224" alt="554270060-0032ddc1-5279-4f3d-9a61-e1d39efd74fd" src="https://github.com/user-attachments/assets/1d43d7ad-9102-4875-a2be-71ae6d2c6c17" />










# RESULT:
   Thus , the implementation of feature scaling and feature selection is completed successfully
