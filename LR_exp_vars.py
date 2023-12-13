import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#df3=pd.read_csv('E1_2207_1184.csv')
#d=pd.read_csv('Ex2_note.csv')
d=pd.read_csv('data/note_embeddings_ex22.csv')
df1=pd.read_csv('Non_notes.csv')
df2=pd.read_csv('BIJAYA_POS.csv')
df=pd.concat([df1,df2],axis=0)
df=df.reset_index(drop=True)
print(len(df))
print(len(d))
df3=pd.concat([df,d],axis=1)
from sklearn import preprocessing
  
# label_encoder object knows 
# how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df3['GENDER']= label_encoder.fit_transform(df3['GENDER'])
#df3=df3['GENDER'].apply(LabelEncoder().fit_transform)
df3=df3.dropna()
print(len(df3))
#d=d.drop(['ddate','adate','pid','label'],axis=1)
#colu=d.columns.tolist()
#df3=df.drop(colu,axis=1)
df3=df3.drop(['CONCATENATED_NOTES'],axis=1) 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
skf = StratifiedKFold(n_splits=3)
y=df3['LABEL']
print(df3.columns)
X=df3.drop(['LABEL'],axis=1).values
n_repetition=30
clf = LogisticRegression(solver='liblinear',multi_class='ovr', max_iter=1000)
#clf = SVC(gamma='auto',probability=True) #RandomForestClassifier(n_estimators=1000, max_depth=2)
eval_results = {"train_auc":[], "test_auc":[], "test_fpr":[], "test_tpr":[],'preds':[]}
feature_importances = []
#undersample = RandomUnderSampler(sampling_strategy='majority')
for rep in range(n_repetition):
    for train_idx, test_idx in skf.split(X,y):
        X_train, X_test = X[train_idx], X[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]
        #X_train, y_train = undersample.fit_resample(X_train, y_train)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        feature_importances.append(clf.coef_[0])
        pred_prob = clf.predict_proba(X_test)
        
        probs = pred_prob[:,clf.classes_ == True].flatten()
        eval_results["preds"].append(probs)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
        auc = metrics.auc(fpr, tpr)
        eval_results["test_auc"].append(auc)
        eval_results["test_fpr"].append(fpr)
        eval_results["test_tpr"].append(tpr)
        train_pred = clf.predict(X_train)
        train_pred_prob = clf.predict_proba(X_train)
        train_probs = train_pred_prob[:,clf.classes_ == True].flatten()
        fpr, tpr, thresholds = metrics.roc_curve(y_train, train_probs)
        acc = metrics.accuracy_score(y_test, pred)
        auc = metrics.auc(fpr, tpr)
        eval_results["train_auc"].append(acc)
print(np.std(eval_results['test_auc']))
print(np.mean(eval_results['test_auc']))

# Calculate mean and standard deviation of feature importances
feature_importances = np.mean(feature_importances, axis=0)
feature_importance_std = np.std(feature_importances)

# Print or visualize feature importances
#print("Mean Feature Importances:")
#print(feature_importances)
#print("Feature Importance Std Deviation:")
#print(feature_importance_std)
df45=pd.DataFrame()
p=df3.columns.tolist()
p.remove('LABEL')
df45['feature']=p
df45['importance']=feature_importances

df65=df45.sort_values(by=['importance'],ascending=False,inplace=False)
df65['feature']=df45['feature']
df65['Odds Ratio']=np.exp(df65['importance'].to_numpy())
df65.to_csv('feature_importance_Domain_notes.csv',index=False)
# Plot feature importances
df55=df65.sort_values(by=['importance'],ascending=False,inplace=False).head(40)
plt.figure(figsize=(10, 6))
plt.bar(range(len(df55)), df55['importance'], tick_label=df55['feature'])
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances of Top 40 features")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("feature_importances_notes.png")
d=pd.read_csv('data/note_embeddings_ex22.csv')
df1=pd.read_csv('Non_notes.csv')
df2=pd.read_csv('BIJAYA_POS.csv')
df=pd.concat([df1,df2],axis=0)
df=df.reset_index(drop=True)
lab=df['LABEL'].tolist()
print(len(df))
print(len(d))
df3=pd.concat([df,d],axis=1)
df3=df3.drop((df.columns.tolist()),axis=1)
df3['LABEL']=lab

df3=df3.dropna()
print(len(df3))
#d=d.drop(['ddate','adate','pid','label'],axis=1)
#colu=d.columns.tolist()
#df3=df.drop(colu,axis=1)
#df3=df3.drop(['CONCATENATED_NOTES'],axis=1) 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
skf = StratifiedKFold(n_splits=3)
y=df3['LABEL']
print(df3.columns)
X=df3.drop(['LABEL'],axis=1).values
n_repetition=30
clf = LogisticRegression(solver='liblinear',multi_class='ovr', max_iter=1000)
#clf = SVC(gamma='auto',probability=True) #RandomForestClassifier(n_estimators=1000, max_depth=2)
eval_results = {"train_auc":[], "test_auc":[], "test_fpr":[], "test_tpr":[]}
feature_importances = []
#undersample = RandomUnderSampler(sampling_strategy='majority')
for rep in range(n_repetition):
    for train_idx, test_idx in skf.split(X,y):
        X_train, X_test = X[train_idx], X[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]
        #X_train, y_train = undersample.fit_resample(X_train, y_train)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        feature_importances.append(clf.coef_[0])
        pred_prob = clf.predict_proba(X_test)
        probs = pred_prob[:,clf.classes_ == True].flatten()
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
        auc = metrics.auc(fpr, tpr)
        eval_results["test_auc"].append(auc)
        eval_results["test_fpr"].append(fpr)
        eval_results["test_tpr"].append(tpr)
        train_pred = clf.predict(X_train)
        train_pred_prob = clf.predict_proba(X_train)
        train_probs = train_pred_prob[:,clf.classes_ == True].flatten()
        fpr, tpr, thresholds = metrics.roc_curve(y_train, train_probs)
        acc = metrics.accuracy_score(y_test, pred)
        auc = metrics.auc(fpr, tpr)
        eval_results["train_auc"].append(acc)
print(np.std(eval_results['test_auc']))
print(np.mean(eval_results['test_auc']))

# Calculate mean and standard deviation of feature importances
feature_importances = np.mean(feature_importances, axis=0)
feature_importance_std = np.std(feature_importances)

# Print or visualize feature importances
#print("Mean Feature Importances:")
#print(feature_importances)
#print("Feature Importance Std Deviation:")
#print(feature_importance_std)
df45=pd.DataFrame()
p=df3.columns.tolist()
p.remove('LABEL')
df45['feature']=p
df45['importance']=feature_importances

df65=df45.sort_values(by=['importance'],ascending=False,inplace=False)
df65['feature']=df45['feature']
df65['Odds Ratio']=np.exp(df65['importance'].to_numpy())
df65.to_csv('feature_importance_onlynotes.csv',index=False)
# Plot feature importances
df55=df65.sort_values(by=['importance'],ascending=False,inplace=False).head(40)
plt.figure(figsize=(10, 6))
plt.bar(range(len(df55)), df55['importance'], tick_label=df55['feature'])
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances of Top 40 features")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("feature_importances_onlynotes.png")
df3=pd.concat([df1,df2],axis=0)
df3=df3.reset_index(drop=True)
from sklearn import preprocessing
  
# label_encoder object knows 
# how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df3['GENDER']= label_encoder.fit_transform(df3['GENDER'])
#df3=df3['GENDER'].apply(LabelEncoder().fit_transform)
df3=df3.dropna()
print(len(df3))
#d=d.drop(['ddate','adate','pid','label'],axis=1)
#colu=d.columns.tolist()
#df3=df.drop(colu,axis=1)
df3=df3.drop(['CONCATENATED_NOTES'],axis=1) 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
skf = StratifiedKFold(n_splits=3)
y=df3['LABEL']
print(df3.columns)
X=df3.drop(['LABEL'],axis=1).values
n_repetition=30
clf = LogisticRegression(solver='liblinear',multi_class='ovr', max_iter=1000)
#clf = SVC(gamma='auto',probability=True) #RandomForestClassifier(n_estimators=1000, max_depth=2)
eval_results = {"train_auc":[], "test_auc":[], "test_fpr":[], "test_tpr":[]}
feature_importances = []
#undersample = RandomUnderSampler(sampling_strategy='majority')
for rep in range(n_repetition):
    for train_idx, test_idx in skf.split(X,y):
        X_train, X_test = X[train_idx], X[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]
        #X_train, y_train = undersample.fit_resample(X_train, y_train)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        feature_importances.append(clf.coef_[0])
        pred_prob = clf.predict_proba(X_test)
        probs = pred_prob[:,clf.classes_ == True].flatten()
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
        auc = metrics.auc(fpr, tpr)
        eval_results["test_auc"].append(auc)
        eval_results["test_fpr"].append(fpr)
        eval_results["test_tpr"].append(tpr)
        train_pred = clf.predict(X_train)
        train_pred_prob = clf.predict_proba(X_train)
        train_probs = train_pred_prob[:,clf.classes_ == True].flatten()
        fpr, tpr, thresholds = metrics.roc_curve(y_train, train_probs)
        acc = metrics.accuracy_score(y_test, pred)
        auc = metrics.auc(fpr, tpr)
        eval_results["train_auc"].append(acc)
print(np.std(eval_results['test_auc']))
print(np.mean(eval_results['test_auc']))

# Calculate mean and standard deviation of feature importances
feature_importances = np.mean(feature_importances, axis=0)
feature_importance_std = np.std(feature_importances)

# Print or visualize feature importances
#print("Mean Feature Importances:")
#print(feature_importances)
#print("Feature Importance Std Deviation:")
#print(feature_importance_std)
df45=pd.DataFrame()
df45['feature']=df3.columns[:-1]
df45['importance']=feature_importances

df65=df45.sort_values(by=['importance'],ascending=False,inplace=False)
df65['Odds Ratio']=np.exp(df65['importance'].to_numpy())
df65.to_csv('feature_importance_domain.csv',index=False)
# Plot feature importances
df55=df65.sort_values(by=['importance'],ascending=False,inplace=False)
plt.figure(figsize=(10, 6))
plt.bar(range(len(df55)), df55['importance'], tick_label=df55['feature'])
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances of Top 20 features")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("feature_importances.png")
