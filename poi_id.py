#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RandomizedLasso

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

my_dataset = data_dict
### Task 2: Remove outliers
my_dataset.pop('TOTAL', 0) #unnecessary spreadsheet data, not a person
my_dataset.pop('THE TRAVEL AGENCY IN THE PARK', 0) #not a person, mostly NaNs

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


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
            

features_list = ['poi','salary', 'total_payments','bonus', \
'total_stock_value', 'expenses', 'other', 'restricted_stock','to_messages', \
'from_poi_to_this_person', 'from_messages', \
'from_this_person_to_poi', 'shared_receipt_with_poi','from_poi_fraction','to_poi_fraction',\
'tot_to_salary','tot_to_bonus','restr_to_total']

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
#Stability Selection:
#http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
rlasso = RandomizedLasso(random_state=2)
rlasso.fit(features,labels)
scores = rlasso.scores_
print scores

for j in range(len(scores)):
    print features_list[j+1],": ",scores[j]
    
features_list_selected = ['poi']
for j in np.where(scores > 0.24)[0]:
    features_list_selected.append(features_list[j+1])


data = featureFormat(my_dataset, features_list_selected)
labels, features = targetFeatureSplit(data)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

clf = DecisionTreeClassifier(min_samples_split = 10,max_features=5,random_state=2)
#clf = RandomForestClassifier(min_samples_split = 4, n_estimators = 1,max_features=4,random_state=2)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
features_list = features_list_selected
test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)