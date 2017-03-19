#########################################################
#														#
#	Text Classification using Naive Bayes  				#
#	Kuldeep Sharma (kuldeepsharma1312@gmail.com)		#
#	Mechanical Engineer @ IIT Delhi						#
#	Github -- @Kuldeep_Attri							#
#														#
#########################################################



import numpy as np
import scipy as sp
import sklearn

###########################################################################
						# Loading data	

results = []
with open('inputfile.txt') as inputfile:
    for line in inputfile:
        results.append(line.strip().split('\t'))

tag =[];data=[];
for i in range(len(results)):
	if(len(results[i])>1):
		if(results[i][0]=="student"):
			tag.append(0)
			data.append(results[i][1])
		elif(results[i][0]=="faculty"):
			tag.append(1)
			data.append(results[i][1])
		elif(results[i][0]=="project"):
			tag.append(2)
			data.append(results[i][1])
		elif(results[i][0]=="course"):
			tag.append(3)
			data.append(results[i][1])			
	
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
train_data=vectorizer.fit_transform(data)
k=4;
m ,d = train_data.shape
label = np.zeros((k,m))
for i in range(m):
	label[tag[i],i]=1
weights=np.zeros((k,d)); # weights[i,j]= prob. of jth word of dict. belongs to ith class
prior=np.zeros((k,1)); # prior[i] = prob. of being ith class


################################################################################
						# Divideing the data in 2 part 

data = train_data[:1500,:]
test_data = train_data[1500:,:]
labels = label[:,:1500]
test_label = label[:,1500:]

#################################################################################
						# Trinaing the Model

weights = ((labels*data)+1)/(np.reshape(np.sum(labels,axis=1)+k,[k,1]));
prior = (np.sum(labels,axis=1)/m);

#################################################################################
						# Testing and calculating accuracy
count=0;
for i_i in range(test_label.shape[1]):

	temp = weights*np.transpose(test_data[i_i,:])
	prob=np.zeros((k,1))
	for i in range(4):
		prob[i,0] = prior[i]*temp[i] 
	if((test_label[np.argmax(prob),i_i]==1.0)):
		count+=1

print "accuracy is...  ",(count/1285.0)*100," %" 	

###################################################################################
