# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:07:12 2015

@author: OlafWied

A first analysis of the ENRON DATASET.

The code tries to follow the project rubric.
"""

import sys
import pprint
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score,recall_score
from sklearn.linear_model import RandomizedLasso
from sklearn.preprocessing import MinMaxScaler

### Load the dictionary containing the dataset


data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

###############################################################################
## UNDERSTANDING THE DATASET AND QUESTION #####################################
###############################################################################

print "-------------DATA EXPLORATION:-------------"

p = 0
total = 0
for name in data_dict:
    if data_dict[name]['poi']:
        p += 1
    total += 1

print "There are ",p," persons of interest."
print "The total number of data points is ",total,"."
print "Each data point consists of ",len(data_dict[name])-1," features"
print "besides 'poi'."
print "We don't include the email address in our analysis."

### Extract features and labels from dataset for local testing
#We don't inlcude the email address as a feature
features_list = ['poi','salary', 'deferral_payments', 'total_payments', \
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', \
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', \
'long_term_incentive', 'restricted_stock', 'director_fees','to_messages', \
'from_poi_to_this_person', 'from_messages', \
'from_this_person_to_poi', 'shared_receipt_with_poi']

#Convert to numpy array
data = featureFormat(data_dict, features_list, sort_keys = True,remove_NaN = False)

data_poi = data[data[:,0]==1]
data_n_poi = data[data[:,0]==0]

print "feature:  missing values (poi)"
for j in range(len(features_list)):
    s = 0
    s_poi = 0
    for k in range(145):
        if math.isnan(data[k,j]):
            s += 1
            if data[k,0]:
                s_poi += 1
    print str(j)+"  "+features_list[j]+": "+ str(s)+" ("+str(s_poi)+")"
    
print "We see that 6 and 14 contain missing values for all pois. No values for pois are \
missing for total_payments, total_stock_value, expenses and 'other'."


print "-------------OUTLIER INVESTIGATION:---------------"

def plot_features(n1,n2):
    print features_list[n1], "vs ", features_list[n2]
    plt.figure()
    plt.plot(data_poi[:,n1],data_poi[:,n2],'ro',alpha=0.5)
    plt.plot(data_n_poi[:,n1],data_n_poi[:,n2],'bo',alpha=0.5)
    plt.show()
    
    
plot_features(1,5) #salary vs bonus

for name in data_dict:
    if data_dict[name]['salary']!= 'NaN' and data_dict[name]['salary'] > 5000000:
        print name, "received a salary above 5,000,000."
    if data_dict[name]['bonus']!= 'NaN' and data_dict[name]['bonus'] > 8000000:
        print name, "received a bonus above 8,000,000."  

print "Delete data point 'TOTAL' because it is a spreadsheet error. Not a person."
#Delete outlier:..................#
data_dict.pop('TOTAL',0)
data = data[data[:,3] < 5000000]
data_poi = data[data[:,0]==1]
data_n_poi = data[data[:,0]==0]
#.................................#

#look for more spreadsheet errors:
#print data_dict.keys() #uncomment to see all names in the dataset
print data_dict['THE TRAVEL AGENCY IN THE PARK']
#Doesn't look like a person => delete
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0) 

plot_features(8,3) #total_stock_value vs total_payments

for name in data_dict:
    if data_dict[name]['total_payments']!= 'NaN' and data_dict[name]['total_payments'] > 100000000:
        print name, " received total payments over 100,000,000."

plot_features(3,9)

#Reload data without outliers
data = featureFormat(data_dict, features_list, sort_keys = True,remove_NaN = False)
#Quick an dirty visualization of the features using boxplots:
#for k in range(1,len(features_list)):
#    print features_list[k]
#    x = data[:,k]
#    z = np.where(np.isnan(x),np.nanmean(x),x)
#    plt.boxplot(z)
#    plt.show()

#Other 'extreme' values seem still plausible considering the amounts of money involved in the scandal
#Huge amounts are usually found for people in the executive committee,
#Therefore, no further outliers are deleted.

###############################################################################
## FEATURE ENGINEERING AND SELECTION ##########################################
###############################################################################

print "-------------CREATE NEW FEATURES:---------------"

print "-------------Old features"
plot_features(16,18) #from poi vs to poi

#Doesn't really show a pattern. We scale the features as follows:

for name in data_dict:
    data_point = data_dict[name]
    
    from_m = data_point["from_messages"]
    to_m = data_point["to_messages"]
    from_poi = data_point["from_poi_to_this_person"]
    to_poi = data_point["from_this_person_to_poi"]
    
    if from_poi == 'NaN' or to_m == 'NaN':
        data_point["from_poi_fraction"] = 0
    else:
        data_point["from_poi_fraction"] = \
            float(from_poi) / to_m
    
    if to_poi == 'NaN' or from_m == 'NaN':
        data_point["to_poi_fraction"] = 0
    else:
        data_point["to_poi_fraction"] = \
            float(to_poi) / from_m
            
            
for name in data_dict:
    data_point = data_dict[name]
    
    bonus = data_point["bonus"]
    tot = data_point["total_payments"]
    salary = data_point["salary"]

    if tot == 'NaN' or salary == 'NaN':
        data_point["tot_to_salary"] = 0
    else:
        data_point["tot_to_salary"] = \
            float(tot) / salary
    
    if tot == 'NaN' or bonus == 'NaN':
        data_point["tot_to_bonus"] = 0
    else:
        data_point["tot_to_bonus"] = \
            float(tot) / bonus       

for name in data_dict:
    data_point = data_dict[name]
    
    bonus = data_point["bonus"]
    tot = data_point["total_payments"]
    salary = data_point["salary"]

    if tot == 'NaN' or salary == 'NaN':
        data_point["tot_to_salary"] = 0
    else:
        data_point["tot_to_salary"] = \
            float(tot) / salary
    
    if tot == 'NaN' or bonus == 'NaN':
        data_point["tot_to_bonus"] = 0
    else:
        data_point["tot_to_bonus"] = \
            float(tot) / bonus
            
for name in data_dict:
    data_point = data_dict[name]
    
    restr = data_point["restricted_stock"]
    tot = data_point["total_stock_value"]
    
    if tot == 'NaN' or restr == 'NaN':
        data_point["restr_to_total"] = 0
    else:
        data_point["restr_to_total"] = \
            float(restr) / tot
            

features_list = ['poi','salary', 'deferral_payments', 'total_payments', \
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', \
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', \
'long_term_incentive', 'restricted_stock', 'director_fees','to_messages', \
'from_poi_to_this_person', 'from_messages', \
'from_this_person_to_poi', 'shared_receipt_with_poi','from_poi_fraction','to_poi_fraction',\
'tot_to_salary','tot_to_bonus','restr_to_total']

data = featureFormat(data_dict, features_list)
data_poi = data[data[:,0]==1]
data_n_poi = data[data[:,0]==0]
print "-------------New features"
plot_features(20,21)
#Looks a little better, for low values of to_poi_fraction there are no pois!


plot_features(22,5)
plot_features(23,1)

print "People with unusual payments to bonus ratio:"
for name in data_dict:
    if data_dict[name]['tot_to_salary']!= 'NaN' and data_dict[name]['tot_to_salary'] > 50:
        print name," total_payments: ",data_dict[name]['total_payments'],", salary: ",data_dict[name]['salary'],"poi:",data_dict[name]['poi']
        
data_dict2 = data_dict.copy()
data_dict2.pop('BANNANTINE JAMES M', 0)
data = featureFormat(data_dict2, features_list)
data_poi = data[data[:,0]==1]
data_n_poi = data[data[:,0]==0]
plot_features(22,5)

plot_features(24,1)





print "-------------SELECT FEATURES:-------------"
print "Stability (Feature) Selction:"
print "(We discard features with too many missing values for Persons of Interest)"
#i.e. deferral_payments, loan_advances, restricted_stock_deferred and director_fees
features_list = ['poi','salary', 'total_payments','bonus','deferred_income', \
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', \
'long_term_incentive', 'restricted_stock','to_messages', \
'from_poi_to_this_person', 'from_messages', \
'from_this_person_to_poi', 'shared_receipt_with_poi','from_poi_fraction','to_poi_fraction',\
'tot_to_salary','tot_to_bonus','restr_to_total']

features_list = ['poi','salary', 'total_payments','bonus', \
'total_stock_value', 'expenses', 'other', 'restricted_stock','to_messages', \
'from_poi_to_this_person', 'from_messages', \
'from_this_person_to_poi', 'shared_receipt_with_poi','from_poi_fraction','to_poi_fraction',\
'tot_to_salary','tot_to_bonus','restr_to_total']
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

#SCALE FEATURES:
#For RandomForest and DecisionTree, scaling is not necessary. 

#scaler = MinMaxScaler()
#features = scaler.fit_transform(features)


#Stability Selection:
#http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
rlasso = RandomizedLasso(random_state=2)
rlasso.fit(features,labels)
scores = rlasso.scores_
print scores

for j in range(len(scores)):
    print features_list[j+1],": ",scores[j]
    
features_list_selected = ['poi']
for j in np.where(scores > 0.3)[0]:
    features_list_selected.append(features_list[j+1])


print "-------------Selected features:-------------"
print features_list_selected

data = featureFormat(data_dict, features_list_selected)
labels, features = targetFeatureSplit(data)

###############################################################################
## PICK AND TUNE ALGORITHM ####################################################
###############################################################################

#PICK AN ALGORITHM 1
rf = RandomForestClassifier(random_state = 3)


#TUNE THE ALGORITHM 1
print "------------Tune Random Forest Classifier:------------"

#Set up a grid search
param_grid = {'min_samples_split':[2,3,4,5],
              'n_estimators':[1,2,3,4,5,7,12,20],
                'max_features':range(len(features_list_selected)/2-1,len(features_list_selected))}
clf = GridSearchCV(rf,param_grid = param_grid,scoring = 'recall')

cv = StratifiedShuffleSplit(labels, 100,test_size = 0.1, random_state = 2)

###############################################################################
## VALIDATE AND EVALUATE ######################################################
###############################################################################

scores = {'m':{},
          'n':{},
          'f': {}}

for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )
    clf.fit(features_train,labels_train)
    p1 =  precision_score(clf.predict(features_test),labels_test)
    p2 = recall_score(clf.predict(features_test),labels_test)
    p = p1*p2
    best = clf.best_estimator_

    m = best.get_params()['min_samples_split']
    n = best.get_params()['n_estimators']
    f = best.get_params()['max_features']
    
    #print p,": ",m,n
    if m in scores['m']:
        scores['m'][m] += p
    else:
        scores['m'][m] = p
    if n in scores['n']:
        scores['n'][n] += p
    else:
        scores['n'][n] = p
    if f in scores['f']:
        scores['f'][f] += p
    else:
        scores['f'][f] = p
    if (m,n,f) in scores:
        scores[(m,n,f)] += p
    else:
        scores[(m,n,f)] = p
    
pprint.pprint(scores)



print "Random Forest Evaluation"
clf = RandomForestClassifier(min_samples_split = 2, n_estimators = 1,max_features=6,random_state=2)
test_classifier(clf, data_dict, features_list_selected)
print clf.feature_importances_

###############################################################################

########################## TUNING AND EVALUATING 2 ############################

###############################################################################
 
print "Print as best result is achieved for n_estimators = 1, a Decision tree might be more appropriate"

#PICK AN ALGORITHM 2
dt = DecisionTreeClassifier(random_state = 2)


#TUNE THE ALGORITHM 1
print "------------Tune Decision Tree Classifier:------------"

#Set up a grid search
param_grid = {'min_samples_split':range(2,len(features_list_selected)),
                'max_features':range(len(features_list_selected)/2-1,len(features_list_selected))}
clf = GridSearchCV(dt,param_grid = param_grid,scoring = 'recall')

cv = StratifiedShuffleSplit(labels, 100 ,test_size = 0.1, random_state = 3)

scores = {'m':{},
          'f': {}}

for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )
    clf.fit(features_train,labels_train)
    p1 =  precision_score(clf.predict(features_test),labels_test)
    p2 = recall_score(clf.predict(features_test),labels_test)
    p = p1*p2
    best = clf.best_estimator_

    m = best.get_params()['min_samples_split']
    f = best.get_params()['max_features']

    if m in scores['m']:
        scores['m'][m] += p
    else:
        scores['m'][m] = p
    if f in scores['f']:
        scores['f'][f] += p
    else:
        scores['f'][f] = p
    if (m,f) in scores:
        scores[(m,f)] += p
    else:
        scores[(m,f)] = p
    
pprint.pprint(scores)

print "Decision Tree Evaluation:"
clf = DecisionTreeClassifier(min_samples_split = 4,max_features=6,random_state=2)
test_classifier(clf, data_dict, features_list_selected)
print clf.feature_importances_