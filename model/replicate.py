# %% [markdown]
# ### Importing Dependies

# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# %% [markdown]
# ### Reading train data

# %% [code]
df=pd.read_csv("../input/malware-detection-va/train.csv",low_memory=False)
y=df['HasDetections']
df.drop('HasDetections',axis=1,inplace=True)

# %% [code]
# df["Census_ThresholdOptIn"].value_counts()

# %% [markdown]
# ### Reading test data

# %% [code]
test=pd.read_csv("../input/malware-detection-va/test_to_give.csv",low_memory=False)

# %% [markdown]
# ### Merging test and train data for consistent preprocessing

# %% [code]
maldata=pd.concat([test.assign(ind="test"), df.assign(ind="train")],axis=0,ignore_index=True)

# %% [code]
# df["HasDetctions"]

# %% [code]
# maldata

# %% [code]
# maldata.head()

# %% [code]
# maldata.shape

# %% [code]
# print(len(maldata.columns))

# %% [markdown]
# ### Observing null values in dataset

# %% [code]
tp=pd.DataFrame((maldata.isnull().sum()/len(maldata))*100).rename(columns={0:"null values"})
tp[tp["null values"]>65]

# %% [markdown]
# ### Dropping features with more than 70% null values

# %% [code]
maldata.drop(["PuaMode","Census_ProcessorClass","DefaultBrowsersIdentifier","Census_IsFlightingInternal","Census_InternalBatteryType"],axis=1,inplace=True)

# %% [markdown]
# ### Dropping features with highly imbalanced value_counts()

# %% [code]
maldata=maldata.drop(["AutoSampleOptIn","Census_IsFlightsDisabled","Census_IsPortableOperatingSystem","SMode","IsBeta"],axis=1)

# %% [code]
# maldata.isnull().sum().sort_values(ascending = False)[:43]

# %% [code]
# maldata.skew().sort_values(ascending = False)[:43]

# %% [code]
# maldata.drop("MachineIdentifier",axis=1,inplace=True)

# %% [code]
# percent = (maldata.isnull().sum()/maldata.shape[0]) * 100
# new_train_data= pd.DataFrame(data=percent,columns=['nullvaluesPercentage'])
# new_train_data = new_train_data.sort_values(by='nullvaluesPercentage',ascending=False)
# print(new_train_data.head(40))

# %% [code]
# for i in maldata.columns:
#     print(i)
#     print(maldata[i].value_counts())

# %% [code]
# maldata.Census_IsWIMBootEnabled.value_counts()

# %% [code]
# maldata.drop("Census_IsWIMBootEnabled",axis=1,inplace=True)

# %% [code]
# percent = (maldata.isnull().sum()/maldata.shape[0]) * 100
# new_train_data= pd.DataFrame(data=percent,columns=['nullvaluesPercentage'])
# new_train_data = new_train_data.sort_values(by='nullvaluesPercentage',ascending=False)
# print(new_train_data.head(15))

# %% [code]
# maldata.OrganizationIdentifier.value_counts()

# %% [code]
maldata.Census_ThresholdOptIn.value_counts()

# %% [code]
maldata.drop('Census_ThresholdOptIn',axis=1,inplace=True)

# %% [code]
maldata['SmartScreen'].value_counts()

# %% [markdown]
# ### This function handles the duplicate upper and lower case values in the smartscreen feature

# %% [code]

def handle_screen(val):
    ''' cleaning category values to reduce number of categories for smartscreen feature '''
    val=str(val)
    if val == 'Block':
        return 'Block'
    elif val == 'ExistsNotSet':
        return 'ExistNotSet'
    elif val == 'Off':
        return 'Off'
    elif val == 'off':
        return 'Off'
    elif val == 'Prompt':
        return 'Prompt'
    elif val == 'prompt':
        return 'Prompt'
    elif val == 'RequireAdmin':
        return 'RequireAdmin'
    elif val == 'requireadmin':
        return 'RequireAdmin'
    elif val == 'Warn':
        return 'Warn'
    elif val == 'On':
        return 'On'
    elif val == 'on':
        return 'On'
    else:
        return 'Unknown'

# %% [code]


# %% [code]
# maldata.SmartScreen.isnull().sum()

# %% [code]
maldata['SmartScreen']=maldata['SmartScreen'].apply(handle_screen)

# %% [code]
# maldata['SmartScreen'].isnull().sum()

# %% [code]
# maldata['SmartScreen'].value_counts()

# %% [code]
# percent = (maldata.isnull().sum()/maldata.shape[0]) * 100
# new_train_data= pd.DataFrame(data=percent,columns=['nullvaluesPercentage'])
# new_train_data = new_train_data.sort_values(by='nullvaluesPercentage',ascending=False)
# print(new_train_data.head(15))

# %% [code]
# maldata['OrganizationIdentifier'].value_counts()

# %% [markdown]
# ### Filling remaining null values with mode

# %% [code]
for feature in maldata.columns:
        if feature=="Census_IsWIMBootEnabled":
            maldata[feature] = maldata[feature].fillna(1)
        else:
            maldata[feature] = maldata[feature].fillna((maldata[feature].mode()[0]))

# %% [code]
# percent = (maldata.isnull().sum()/maldata.shape[0]) * 100
# new_train_data= pd.DataFrame(data=percent,columns=['nullvaluesPercentage'])
# new_train_data = new_train_data.sort_values(by='nullvaluesPercentage',ascending=False)
# print(new_train_data.head(15))

# %% [code]


# %% [markdown]
# ### Extracting object type columns

# %% [code]
cat=[]
for i in maldata.columns:
    if (maldata[i].dtype=="object" and i!='ind' and i!='MachineIdentifier'):
        cat.append(i)
cat

# %% [code]
len(cat)

# %% [markdown]
# ### Replacing values with low count with mode in ProductName

# %% [code]
def replace_ProductName(val):
    
    if val=="win8defender" or val=="scep" or val=="windowsintune" or val=="fep":
        return "win8defender"
    else:
        return val

# %% [code]
maldata["ProductName"]=maldata["ProductName"].apply(replace_ProductName)

# %% [code]
# def replace_EngineVersion(val):
#     if val=="1.1.11502.0":
#         return "1.1.15200.1"
#     else:
#         return val

# %% [code]
# def replace_ChasisType(val):
#     if val.isnumeric() or val=="ExpansionChassis" or val=="DockingStation" or val=="CompactPCI":
#         return "Notebook"
    
#     else:
#         return val

# %% [code]
# def replace_OSEdition(val):
#     if val[-1]=="N":
#         val=val[:-1]
#     if val[-1]=="S":
#         val=val[:-1]
#     elif val=="Home" or val=="Pro" or  val=="ServerDatacenterACor" or val=="ProfessionalSingleLanguage":
#         val="Core"
#     else:
#         val=val
#     return val

# %% [code]
# maldata["EngineVersion"]=maldata["EngineVersion"].apply(replace_EngineVersion)

# %% [code]
# maldata["Census_ChassisTypeName"]=maldata["Census_ChassisTypeName"].apply(replace_ChasisType)

# %% [code]
# maldata["Census_OSEdition"]=maldata["Census_OSEdition"].apply(replace_OSEdition)

# %% [code]
# maldata["Census_OSEdition"].value_counts()

# %% [code]
# j=filter(fun,(maldata["AvSigVersion"].value_counts().tolist()))
# sum=0
# for i in j:
#     sum+=1
# sum

# %% [markdown]
# ### Separating categorical into two lists, yo and cat, for label and one hotencoding respectively

# %% [markdown]
# #### One Hot encoding was resulting in exceeded RAM, so we label encoded columns with high dimensionality

# %% [code]
yo=["AvSigVersion","OsBuildLab","Census_OSVersion","AppVersion","Census_OSSkuName","ProductName","EngineVersion","Census_ChassisTypeName","Census_OSEdition"]

# %% [code]
cat.remove("AvSigVersion")
cat.remove("OsBuildLab")
cat.remove("Census_OSVersion")
cat.remove("ProductName")
cat.remove("EngineVersion")
cat.remove("AppVersion")
cat.remove("Census_ChassisTypeName")
cat.remove("Census_OSEdition")
cat.remove("Census_OSSkuName")

# %% [markdown]
# ### Applying encoding and obtaining preprocessed dataset

# %% [code]
l=maldata.drop(cat+yo,axis=1)

# %% [code]
l

# %% [code]
len(cat)

# %% [code]
# cat

# %% [code]
# len(cat)

# %% [code]
# h=maldata[cat]

# %% [code]
# h

# %% [code]
# h["ProductName"]

# %% [code]
# h["ProductName"].nunique()

# %% [code]
# def tp(h):
    
#     for i in h.columns:
#         index=h[h.groupby(i)[i].transform('count').lt(2)].index
#         index
# # h["ProductName"].nunique()
#         h.loc[index,i]=np.NaN
#         h[i]=h[i].fillna(h[i].mode()[0])
# #         maldata['ProductName'].nunique()
# # h.loc[h.groupby('ProductName').ProductName.transform('count').lt(2), 'ProductName'] = np.nan  
# # h['ProductName']
# # h["ProductName"].nunique()
#     return h
# # h["ProductName"].nunique()

# %% [code]
# h=tp(h)
# for i in h.columns:
#     print (i)
#     print (h[i].nunique())

# %% [code]
# h.nunique()

# %% [code]
# len(yo)

# %% [markdown]
# ### Applying encoding and obtaining preprocessed dataset

# %% [code]
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
enc=LabelEncoder()
enc1=OneHotEncoder(handle_unknown='ignore',sparse=True)

# %% [code]
cat_tp=pd.DataFrame()
label_tp=pd.DataFrame()

# %% [code]
for i in cat:
#     enc = OneHotEncoder(handle_unknown='ignore',sparse=True)
    enc_df = enc1.fit_transform(maldata[[i]]).toarray()
    names=enc1.get_feature_names([i])
    enc_df=pd.DataFrame(enc_df,columns=names)
    cat_tp=pd.concat([cat_tp,enc_df],axis=1)

# %% [code]
for i in yo:
    labelen=maldata[i]
    labelen=pd.DataFrame(enc.fit_transform(labelen))
    labelen=labelen.rename(columns={0:i})
    label_tp=pd.concat([label_tp,labelen],axis=1)

# %% [code]
l

# %% [code]
label_tp

# %% [code]
cat_tp

# %% [code]
# for i in cat:
#     print (i)
#     print(maldata[i].nunique())

# %% [code]
l=pd.concat([l,cat_tp,label_tp],axis=1)

# %% [code]
l

# %% [markdown]
# ### Separating back into submisssion test data and train data

# %% [code]
test= l[l["ind"].eq("test")].copy()
maldata=l[l["ind"].eq("train")].copy()

# %% [code]
test["ind"]

# %% [code]
# maldata.reset_index(drop=True)

# %% [code]
maldata["ind"]

# %% [code]
test.drop("ind",axis=1,inplace=True)
maldata.drop("ind",axis=1,inplace=True)

# %% [code]
test

# %% [code]
maldata.reset_index(drop=True)

# %% [code]
# for i in l.columns:
#     print(i)

# %% [code]
l=maldata.copy()

# %% [code]
# l.reset_index(drop=True)

# %% [code]
y

# %% [code]
# for i in cat:
#     dumm=pd.get_dummies(maldata[i],prefix=i)
# #     print(l)
#     l=pd.concat([l,dumm],axis=1)

# %% [code]
# y=l['HasDetections']

# %% [markdown]
# ### Dropping MachineIdentifier as it is a unique value for each data entry

# %% [code]
l.drop('MachineIdentifier',axis=1,inplace=True)

# %% [code]
# l

# %% [code]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

# %% [code]
# l.reset_index(drop=True)

# %% [code]
# import re
# l= l.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# %% [markdown]
# ### Splitting train data into train and test sets

# %% [code]
X_train,X_test,Y_train,Y_test=train_test_split(l,y,stratify=y,random_state=2019,test_size=0.2)

# %% [code]
# X_train

# %% [markdown]
# ### Tries Random Forets also.

# %% [code]
# model=RandomForestClassifier(n_estimators=1200, min_samples_leaf=1000, max_leaf_nodes=150, n_jobs=-1,class_weight='balanced_subsample')

# %% [code]
import xgboost as xgb

# %% [code]
# xgb_params = {}
# xgb_params['learning_rate'] = 0.1
# xgb_params['n_estimators'] = 230
# xgb_params['max_depth'] = 12
# xgb_params['num_leaves']=150,
# xgb_params['min_data_in_leaf']:300,

# # xgb_params['subsample'] = 0.9
# xgb_params['colsample_bytree'] = 0.176
# # xgb_params['scale_pos_weight']=4

# %% [code]
from lightgbm import LGBMClassifier

# %% [code]


# %% [code]
# clf = LGBMClassifier(objective='binary',n_jobs=-1,n_estimators=230)
# clf.fit(X_train,Y_train)

# %% [code]
# model.fit(X_train,Y_train)

# %% [code]
# output=model.predict_proba(X_test)[:,1]

# %% [code]
# from sklearn.metrics import roc_auc_score

# %% [code]
# print(2*roc_auc_score(Y_test,output)-1)

# %% [code]
# model=xgb.XGBClassifier(**xgb_params)

# %% [code]
# scaler=StandardScaler()
# X_train=scaler.fit_transform(X_train)
# X_test=scaler.fit_transform(X_test)

# %% [code]
# clf.fit(X_train,Y_train)

# %% [code]
# output=clf.predict_proba(X_test)[:,1]

# %% [code]
# print(2*roc_auc_score(Y_test,output)-1)

# %% [code]
# model.fit(X_train,Y_train)

# %% [code]


# %% [code]


# %% [markdown]
# ### We tuned hyperparameters using GridSearchCV and RandomizedSearchCV

# %% [code]
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# %% [markdown]
# ### Hyperparameters for LightGBM Classifier

# %% [code]
parameters = {
              'learning_rate': 0.1,
              'n_estimators':230,
              'boosting_type':'gbdt',
              'objective':['binary'],
              'colsample_bytree':0.176,
              'num_leaves':150,
              'min_data_in_leaf':300,
              'max_depth':12,
              'n_jobs':[-1],
#               'reg_lambda':0.6,
#               'reg_alpha':3,
                
              }

# %% [code]
# clf = GridSearchCV(model,parameters)

# %% [code]
model= LGBMClassifier(**parameters)

# %% [code]
# model1=LGBMClassifier(**parameters1)

# %% [code]
# model2=LGBMClassifier(**parameters2)

# %% [markdown]
# ### We tried xgboost,RandomForest and logisitic regression also but LighGBM+EasyEnsemble outperformed them.

# %% [code]
# import xgboost as xgb

# %% [code]
# clf=xgb.XGBClassifier(learning_rate=0.1, 
#                             n_estimators=230, 
#                             max_depth=12,
# #                            
#                             colsample_bytree=0.176,
#                             objective= 'binary:logistic',
#                             nthread=-1,
#                             scale_pos_weight=4,
#                             booster='gbtree',
# #                            
#                             tree_method = 'hist',
#                             seed=42)

# %% [code]


# %% [code]
# pd.DataFrame(Y_test).value_counts()

# %% [code]
# Y_test.value_counts()

# %% [code]
# 17032/96514

# %% [markdown]
# ### Using EasyEnsembleClassifier for better score and avoid overfitting.

# %% [code]
from imblearn.ensemble import EasyEnsembleClassifier
clf=EasyEnsembleClassifier(n_estimators=30,base_estimator=model,random_state=42,n_jobs=-1,sampling_strategy='majority',verbose=True)

# %% [code]
# model.fit(X_train,Y_train,eval_set=(X_test,Y_test),eval_metric='auc',verbose=10)

# %% [code]
# clf.fit(X_train,Y_train,eval_set=(X_test,Y_test),eval_metric='auc',verbose=10,early_stopping_rounds=10)

# %% [markdown]
# ### Training the model

# %% [code]
clf.fit(X_train,Y_train)

# %% [markdown]
# ### Making Predictions

# %% [code]
output=clf.predict_proba(X_test)[:,1]

# %% [markdown]
# ### Final training roc_auc score

# %% [code]
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(Y_test, output,pos_label=1)
auc_score=metrics.auc(fpr, tpr)
print (auc_score)

# %% [markdown]
# ### This gave us a final score of 0.71820 on the  private leaderboard

# %% [markdown]
# ### Scores we got during training

# %% [code]
# 0.7168935719666678

# %% [code]
# 0.7167725823085187

# %% [code]
# 0.7165580523619851

# %% [code]
# 0.716907921413368 n_est=500, lr=0.09

# %% [code]
# 0.716430307732827 -

# %% [code]
# for i in test.columns:
#     if i not in l.columns:
#         print(i)

# %% [code]
# 0.7057052859268755 - lightgbm, n_est=230,col=0.176,'gbdt'

# %% [code]
# 0.704918546547196 - loghtgbm, n_est=230

# %% [code]
# 0.7048313925777644 - n_est=250

# %% [code]
# 0.704784706079872 - n_est=200

# %% [code]
# 0.7041714241843127- nestimators=150

# %% [code]
# 0.7028641891640863 - n_est=100

# %% [code]
# 0.7025317936726614

# %% [markdown]
# **Test**

# %% [markdown]
# ### Dropping MachineIdentifier from test also and saving it to use later in submission

# %% [code]
id=test["MachineIdentifier"]

# %% [code]


test.drop("MachineIdentifier",axis=1,inplace=True)

# %% [code]
test

# %% [code]
# def replace_ChasisType1(val):
#     val=str(val)
#     if val.isnumeric():
#         return "Notebook"
#     elif val=="nan":
#         return "UNKNOWN"
#     else:
#         return val

# %% [code]
# test["Census_ChassisTypeName"]=test["Census_ChassisTypeName"].apply(replace_ChasisType1)

# %% [code]
# test['Census_OSEdition']=test['Census_OSEdition'].apply(replace_OSEdition)

# %% [code]
# test=scaler.fit_transform(test)

# %% [markdown]
# ### Predicting on test data and making submission file.

# %% [code]
output_1=clf.predict_proba(test)[:,1]

# %% [code]
submit1=pd.DataFrame(output_1)

# %% [code]
submit1.value_counts()

# %% [code]
submit1=pd.concat([id,submit1],axis=1)

# %% [code]
submit1

# %% [code]
submit1=submit1.rename(columns={0:'HasDetections'})

# %% [code]
submit1

# %% [code]
id

# %% [code]
submit1.to_csv("./assignment_output36.csv",index=False)