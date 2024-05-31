from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC

models = {
    #'CatBoost' : (CatBoostClassifier(silent=True), {'n_estimators': [100, 200, 300]}),
    #'XGBRFClassifier' : (XGBRFClassifier(), {'n_estimators': [50, 100, 200]}),
    'NaiveBayes' : (GaussianNB(), {}),
    'SGD' : (SGDClassifier(), {'alpha' : [0.0001,0.001,0.01]}),
    'KNN' : (KNeighborsClassifier(), {'n_neighbors' : [3,5,7]}),
    'DecsionTree' : (DecisionTreeClassifier(), {'max_depth' : [None,10,20]}),
    'RandomForest' : (RandomForestClassifier(), {'n_estimators' : [50,100,200]}),
    'LogisticRegression' : (LogisticRegression(), {'C' : [0.1,1,10]}),
    'XGBoost' : (XGBClassifier(), {'n_estimators' : [50,100,200]}),
    'AdaBoost': (AdaBoostClassifier(algorithm='SAMME'), {'n_estimators': [50, 100, 200]}),
    'ExtraTrees' : (ExtraTreesClassifier(), {'n_estimators' : [50,100,200]}),
    'LDA' : (LinearDiscriminantAnalysis(), {}),
    'RidgeClassifier' : (RidgeClassifier(), {'alpha' : [0.1,1,10]}),
    'Lasso' : (LogisticRegression(penalty='l1', solver = 'liblinear'), {'C' : [0.1,1,10]}),
    'ElasticNet' : (SGDClassifier(penalty='elasticnet'), {'alpha' : [0.0001,0.001,0.01]}),
    'SVM' : (SVC(), {'C' : [0.1,1,10]}),
    'GBM' : (GradientBoostingClassifier(), {'n_estimators': [50, 100, 200]}),
    'LightGBM' : (LGBMClassifier(verbosity=-1), {'n_estimators': [50, 100, 200]})
}