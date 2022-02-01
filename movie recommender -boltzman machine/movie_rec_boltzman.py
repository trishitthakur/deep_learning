'''from machine learning A-Z course
   coded by trishit nath thakur'''

#Boltzmann Machine
##dataset
###ML-100K
"""

"http://files.grouplens.org/datasets/movielens/ml-100k.zip"

"""

##Importing libraries"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

"""## Importing dataset"""

# We won't be using this dataset it is for 1M dataset on website.
#movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') #some movies contain special character so encoding changed
#users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

"""## Preparing the training set and the test set"""


training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t') 
training_set = np.array(training_set, dtype = 'int') # pytorch tensor need array instead of dataframe so convert to array of integers
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int') # pytorch tensor need array instead of dataframe so convert to array of integers


"""## Getting the number of users and movies"""


nb_users = int(max(max(training_set[:, 0], ), max(test_set[:, 0]))) # maximum number in the user column for both train and test data
nb_movies = int(max(max(training_set[:, 1], ), max(test_set[:, 1]))) # # maximum number in the movie column for both train and test data


"""## Converting the data into an array with users in lines and movies in columns"""


def convert(data):
  
  new_data = []    # final array with user in line and movies in column 
  
  for id_users in range(1, nb_users + 1):
    
    id_movies = data[:, 1] [data[:, 0] == id_users] # list of all movies rated  by the nth user
    
    id_ratings = data[:, 2] [data[:, 0] == id_users]  # list of all ratings given by the nth user
    
    ratings = np.zeros(nb_movies) # we create list of nb_movies zeros 
    
    ratings[id_movies - 1] = id_ratings # replace 0 for the movies rated by person by the rating
    
    new_data.append(list(ratings)) # add this list of all ratings to the new_data
  
  return new_data

training_set = convert(training_set) # training list of lists created

test_set = convert(test_set) # test list of lists created


"""## Converting the data into Torch tensors"""


training_set = torch.FloatTensor(training_set) # tensors are array that contains element of single data type ie multidimensional matrix(type here will be float)
test_set = torch.FloatTensor(test_set) # input in it is a list of lists


"""## Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)"""


training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


"""## Creating the architecture of the Neural Network"""



class RBM():
  
  def __init__(self, nv, nh): # nv is number of visible nodes while nh is number of hidden nodes
    
    self.W = torch.randn(nh, nv) # all weights initilized in a torch tensor(p(hidden nodes) given the visible nodes) in matrix of size nh,nv
                                 # randn to initialise randomly according to a normal distribution with mean = 0 and variance of 1
    
    self.a = torch.randn(1, nh) # bias for p(visible nodes) given the hidden nodes
    
    self.b = torch.randn(1, nv) # bias for p(hidden nodes) given the visible nodes
  
  
  def sample_h(self, x):           # sampling hidden nodes accn to p_h_given_v, x is visible neurons for p_h_given_v
    
    wx = torch.mm(x, self.W.t())         # for approximating log likelihood gradient using gibbs sampling
    
    activation = wx + self.a.expand_as(wx)  # wx is product of vector of weights and vector of neurons while activation is wx + bias
    
    p_h_given_v = torch.sigmoid(activation)  # .expand_as func used so that bias is applied to each line of minibatch we created using (1, nh)
    
    return p_h_given_v, torch.bernoulli(p_h_given_v) # bernoulli rbm for binary outcome
  
  
  def sample_v(self, y):          # sampling visible nodes accn to p_v_given_h, y is hidden neurons for p_v_given_h
    
    wy = torch.mm(y, self.W)     # for approximating log likelihood gradient using gibbs sampling
    
    activation = wy + self.b.expand_as(wy)     # wy is product of vector of weights and vector of neurons while activation is wy + bias
    
    p_v_given_h = torch.sigmoid(activation)   # .expand_as func used so that bias is applied to each line of minibatch we created using (1, nv)
    
    return p_v_given_h, torch.bernoulli(p_v_given_h) # bernoulli rbm for binary outcome
  
  
  def train(self, v0, vk, ph0, phk):       
    
      # vo is input vector, vk is visible nodes after k sampling, ph is vector of probability that at first time hidden node = 1 given value of v0, similarly phk.
    
    self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t() # updating weights
      
    self.b += torch.sum((v0 - vk), 0) #updating bias for hidden nodes
    
    self.a += torch.sum((ph0 - phk), 0) # updating bias for visible nodes


nv = len(training_set[0]) # number of movies
nh = 100
batch_size = 100
rbm = RBM(nv, nh) # rbm object of class RBM



"""## Training the RBM"""



nb_epoch = 10

for epoch in range(1, nb_epoch + 1):
  train_loss = 0 
  s = 0.       # float for a counter to normalise the train loss
  
  
  for id_user in range(0, nb_users - batch_size, batch_size):
    
    vk = training_set[id_user : id_user + batch_size]
    
    v0 = training_set[id_user : id_user + batch_size]
    
    ph0,_ = rbm.sample_h(v0)  # ,_ used to get the first element of what func returns
    
    
    for k in range(10):
      
      _,hk = rbm.sample_h(vk)   # _, used to get the second element of what func returns
      
      _,vk = rbm.sample_v(hk)   # _, used to get the second element of what func returns
      
      vk[v0<0] = v0[v0<0]       # freeze visible nodes that contain minus ratings because they were not rated
    
    phk,_ = rbm.sample_h(vk)
    rbm.train(v0, vk, ph0, phk)
    train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0])) # simple distance between real rating and predicted rating for loss computations
    s += 1.
  
  
  print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))


"""## Testing the RBM"""


test_loss = 0
s = 0.


for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]   # training set is input used to activate neurons to get the output
    vt = test_set[id_user:id_user+1]
    
    
    if len(vt[vt>=0]) > 0:
       
        _,h = rbm.sample_h(v) # we dont need to make 10 steps again because its testing process
       
        _,v = rbm.sample_v(h)
       
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.


print('test loss: '+str(test_loss/s))