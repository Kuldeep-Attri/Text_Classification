#################################################
#	Text Classification using Naive Bayes       
#													
#	Kuldeep Sharma (19/03/2017)					
#	Mechanical Engineer @ IIT Delhi			
#													
#	Github @Kuldeep-Attri						     
#												   
#################################################	

import numpy as np
import scipy as sp
import sklearn

###########################################################################
						# Loading data	

results = []
with open('trainfile.txt') as inputfile:
    for line in inputfile:
        results.append(line.strip().split('\t'))

train_tag =[];train_data=[];
for i in range(len(results)):
	if(len(results[i])>1):
		if(results[i][0]=="student"):
			train_tag.append(0)
			train_data.append(results[i][1])
		elif(results[i][0]=="faculty"):
			train_tag.append(1)
			train_data.append(results[i][1])
		elif(results[i][0]=="project"):
			train_tag.append(2)
			train_data.append(results[i][1])
		elif(results[i][0]=="course"):
			train_tag.append(3)
			train_data.append(results[i][1])	
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
train_data=vectorizer.fit_transform(train_data)

with open('testfile.txt') as inputfile:
    for line in inputfile:
        results.append(line.strip().split('\t'))       

test_tag =[];test_data=[];
for i in range(len(results)):
	if(len(results[i])>1):
		if(results[i][0]=="student"):
			test_tag.append(0)
			test_data.append(results[i][1])
		elif(results[i][0]=="faculty"):
			test_tag.append(1)
			test_data.append(results[i][1])
		elif(results[i][0]=="project"):
			test_tag.append(2)
			test_data.append(results[i][1])
		elif(results[i][0]=="course"):
			test_tag.append(3)
			test_data.append(results[i][1])					
	
k=4; # Number of classes
 
test_data=vectorizer.transform(test_data)

number_of_word = train_data.sum(axis=1)

train_label = np.zeros((k,train_data.shape[0]))
test_label = np.zeros((k,test_data.shape[0]))

vocab_size = train_data.shape[1]


for i in range(train_data.shape[0]):
	train_label[train_tag[i],i]=1
for i in range(test_data.shape[0]):
	test_label[test_tag[i],i]=1

weights=np.zeros((k,train_data.shape[1])); # weights[i,j]= prob. of jth word of dict. belongs to ith class
prior=np.zeros((k,1)); # prior[i] = prob. of being ith class

#################################################################################
						# Trinaing the Model

weights = ((train_label*train_data)+1)/((train_label*number_of_word)+vocab_size);
prior = (np.sum(train_label,axis=0)/train_data.shape[0]);

						# Using In-built Naive Bayes

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_data,train_tag)


#################################################################################
						# Testing and calculating accuracy
count=0;count_in=0;
for i_i in range(test_label.shape[1]):
	temp = np.log(weights)*np.transpose(test_data[i_i,:])
	prob=np.zeros((k,1))
	for i in range(k):
		prob[i,0]=np.log(prior[i]) + (temp[i])
	if((test_label[np.argmax(prob),i_i]==1.0)):
		count+=1
	if((test_label[clf.predict(test_data[i_i,:]),i_i]==1.0)):
		count_in+=1;


print "Accuracy is (using my Implementation of Naive Bayes)...  ",(count/float(test_label.shape[1]))*100," %" 
print "Accusrcy is (using in built Naive Bayes)...  ",(count_in/float(test_label.shape[1]))*100, " %"	

###################################################################################
