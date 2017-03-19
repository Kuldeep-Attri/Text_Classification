#####################################################
#													#	
#	Text Classification (Document Classification)	#
#													#
#	using -- Naive Bayes							#
#													#		
#	Kuldeep Sharma (20/03/2017)						#
#													#	
#	Github --	    								#
#####################################################

import numpy as np 

#def load_data(filename):

	# This is to load data set	

def train_model(data,labels,weights,prior):

	# This is to train my model
	m = data.shape[0];
	d = data.shape[1];
	k = labels.shape[1];

	# This is one way, using for loops
	# for i_ in range(d):
	# 	for i in range(k):
	# 		num=0;
	# 		den=0;
	# 		for j in range(m):
	# 			num = num + labels[i,j]*data[j,i_]; 
	# 			den = den + labels[i,j];
	# 		weights[i,i_] = (num+1)/(den+k);	# Here I am also handling the laplace smoothing	 
	# 		prior[i,1]=den/m;

	# Second way, vectorization in python (Much Faster)		

	weights = (np.dot(labels,data)+1)/(np.reshape(np.sum(labels,axis=1)+k,[k,1]));
	prior = (np.sum(labels,axis=1)/m);

def test_model(input_, weights, prior):

	# This is to test my model on given input

	prob = prior * (np.dot(weights,input_));
	for i in range(weights.shape[0]):
		print prob[i]

	print "Label is: ",np.argmax(prob),"."	

if __name__=="__main__": 

	d=100; # dimension of the features vector (This is also equal to size of dictionary)
	m=1000; # number of training example
	k=5; # number of classes

	# Load Data will be called here once we have dataset
	data = np.random.rand(m,d); # This is my data set for the time being
	labels = np.random.rand(k,m); # Labels of the given dataset

	weights=np.zeros((k,d)); # weights[i,j]= prob. of jth word of dict. belongs to ith class
	prior=np.zeros((k,1)); # prior[i] = prob. of being ith class

	train_model(data,labels,weights,prior);

	input_ = np.random.rand(d,1);

	test_model(input_,weights,prior);




