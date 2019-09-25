# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,Imputer
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

def main():
  # data load
  df = pd.read_csv('./data/'+ file_model + '.csv', header=0)
  ID = df.iloc[:,0] 
  y = df.iloc[:,-1]
  X = df.iloc[:,1:-1]

  # preprocessing-1: one-hot encoding
  X_ohe = [-------](X, dummy_na=True, columns=ohe_cols)
  X_ohe = X_ohe.dropna(axis=1, how='all')
  X_ohe_columns = X_ohe.columns.values

  # preprocessing-2: null imputation
  imp = [-------]
  imp.fit(X_ohe)
  X_ohe = pd.DataFrame([-------], columns=[-------])
  print(X_ohe.shape)

  # preprocessing-3: feature selection
  selector = [-------](estimator=RandomForestClassifier(random_state=0),step=0.05)
  selector.fit(X_ohe, y.as_matrix().ravel())
  X_ohe_selected = selector.transform(X_ohe)
  X_ohe_selected = pd.DataFrame(X_ohe_selected, columns=[-------])
  print(X_ohe_selected.shape)
  X_ohe_selected.head()

  # preprocessing-4: preprocessing of a score data along with a model dataset
  if len(file_score)>0:
      # load score data
      dfs = pd.read_csv('./data/'+ file_score + '.csv', header=0)
      IDs = dfs.iloc[:,[0]] 
      Xs = dfs.iloc[:,1:-1]
      Xs_ohe = pd.get_dummies(Xs, dummy_na=True, columns=ohe_cols)
      cols_m = pd.DataFrame(None, columns=X_ohe_columns, dtype=float)

      # consistent with columns set
      Xs_exp = pd.concat([cols_m, Xs_ohe])
      Xs_exp.loc[:,list(set(X_ohe_columns)-set(Xs_ohe.columns.values))] = \
          Xs_exp.loc[:,list(set(X_ohe_columns)-set(Xs_ohe.columns.values))].fillna(0, axis=1)
      Xs_exp = Xs_exp.drop(list([-------]), axis=1)

      # re-order the score data columns
      Xs_exp = Xs_exp.reindex_axis(X_ohe_columns, axis=1)
      Xs_exp = pd.DataFrame(imp.transform(Xs_exp), columns=X_ohe_columns)
      Xs_exp_selected = Xs_exp.loc[:,[-------]]

  # modeling
  clf.fit(X_ohe_selected, y.as_matrix().ravel())
  joblib.dump(clf, './model/'+ model_name + '.pkl')
  results = cross_val_score(clf, X_ohe_selected, y.as_matrix().ravel(), scoring='roc_auc', cv=5)
  print('cv score:', np.average(results), '+-', np.std(results))

  # scoring
  if len(file_score)>0:
      score = pd.DataFrame(clf.predict_proba(Xs_exp_selected)[:,1], columns=['pred_score'])
      IDs.join(score).to_csv('./data/'+  model_name + '_' + file_score + '_with_pred.csv', index=False)

  # model profile
  imp = pd.DataFrame([clf.named_steps['est'].feature_importances_], columns=X_ohe_columns[selector.support_])
  imp.T.to_csv('./data/'+  model_name + '_feature_importances.csv', index=True)

if __name__ == '__main__':

  # SET PARAMETERS
  file_model = 'dm_for_model'
  file_score = 'dm_for_fwd'
  ohe_cols = ['mode_category']

  # CLASSIFIER
  model_name = 'GBC_001'
  clf = Pipeline([('scl',StandardScaler()), ('est',GradientBoostingClassifier(random_state=1))])

  ######################################
  # GBC_001の処理が終わったらこちらも実行すること
  ######################################
  # model_name = 'RF_001'
  # clf = Pipeline([('scl',StandardScaler()), ('est',RandomForestClassifier(random_state=1))])

  # MAIN PROC
  main()
