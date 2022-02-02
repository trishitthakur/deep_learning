'''from deep learning A-Z course
   coded by trishit nath thakur'''

##Downloading the dataset

###ML-100K
"""

"http://files.grouplens.org/datasets/movielens/ml-100k.zip"


"""###ML-1M"""

"http://files.grouplens.org/datasets/movielens/ml-1m.zip"



"""##Importing libraries"""




import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable



"""## Importing the dataset"""



# We won't be using this dataset.
#movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')




"""## Preparing the training set and the test set"""



training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')      # pytorch tensor need array instead of dataframe so convert to array of integers


test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')    # pytorch tensor need array instead of dataframe so convert to array of integers



"""## Getting the number of users and movies"""



nb_users = int(max(max(training_set[:, 0], ), max(test_set[:, 0])))   # maximum number in the user column for both train and test data
nb_movies = int(max(max(training_set[:, 1], ), max(test_set[:, 1])))   # # maximum number in the movie column for both train and test data 



"""## Converting the data into an array with users in lines and movies in columns"""



def convert(data): 
  
  new_data = []      # final array with user in line and movies in column 
  
  for id_users in range(1, nb_users + 1):
  
    id_movies = data[:, 1] [data[:, 0] == id_users]   # list of all movies rated  by the nth user
  
    id_ratings = data[:, 2] [data[:, 0] == id_users]  # list of all ratings given by the nth user
  
    ratings = np.zeros(nb_movies)     # we create list of nb_movies zeros 
  
    ratings[id_movies - 1] = id_ratings   # replace 0 for the movies rated by person by the rating
  
    new_data.append(list(ratings))   # add this list of all ratings to the new_data
  
  return new_data 


training_set = convert(training_set)   # training list of lists created

test_set = convert(test_set)  # test list of lists created



"""## Converting data into Torch tensors"""



training_set = torch.FloatTensor(training_set) # tensors are array that contains element of single data type ie multidimensional matrix(type here will be float)

test_set = torch.FloatTensor(test_set)  # input in it is a list of lists



"""## architecture of the Neural Network"""



class SAE(nn.Module):  # inherit from module class from nn module
    
    
    def __init__(self, ):   #self, used for variables of module class
        
        super(SAE, self).__init__() # to get inherited classes and methods from module class
        
        self.fc1 = nn.Linear(nb_movies, 20) # first full connection bw input vector of features which is rating of all movies and hidden layer which is shorter vector 
        
        self.fc2 = nn.Linear(20, 10) # second  full connection bw first hidden layer and second hidden layer which we take as 10 neurons
        
        self.fc3 = nn.Linear(10, 20) # we starting to decode ie reconstruct original input vector
        
        self.fc4 = nn.Linear(20, nb_movies) # last full connection that we have to reconstruct input layer so same dimension used as input
        
        self.activation = nn.Sigmoid()
    
    
    def forward(self, x): # for encoding decoding and applying activation functions on fully connected layers  

                          # x is simply the input vector
        
        x = self.activation(self.fc1(x)) # encoding input vector to shorter vector of 20 neurons
        
        x = self.activation(self.fc2(x)) # encoding vector of 20 neurons to further shorter vector of 10 neurons
        
        x = self.activation(self.fc3(x)) # decoding input vector of 10 neurons to output vector of 20 elements
        
        x = self.fc4(x) # no activation function for final layer 
        
        return x # vector of predicted ratings that we will compare to vector of real ratings ie input vector


sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # decay used to reduce the learning rate after epochs to regulate convergence



"""## Training the SAE"""




nb_epoch = 200

or epoch in range(1, nb_epoch + 1):

  train_loss = 0

  s = 0.        # float for a counter to normalise the train loss
  
  
  for id_user in range(nb_users):

    input = Variable(training_set[id_user]).unsqueeze(0)  # network in pytorch cannot take a single vector of 1D. it needs a batch of input vectors
    
     # unsqueeze from variable function for added fake dimension corresponding to batch for predict method to accept, 0 for index of new dim

    target = input.clone()     # target variable is original input vector before the modifications performed                            
    
    
    if torch.sum(target.data > 0) > 0:   # to save memory and take only users who rated at least 1 movie

      output = sae(input)  

      target.require_grad = False    # so that gradient is computed only wrt to target and not the input to save memory

      output[target == 0] = 0    # to include only movies that user rated to save memory

      loss = criterion(output, target)  # compute loss error

      mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # average of error but by only considering movies that were rated

      loss.backward() # do we need to increase the weight or decrease the weights ie sets the direction

      train_loss += np.sqrt(loss.data[0]*mean_corrector) # 0 is index of data that contain train_loss, multiplied to get adjusted mean, root taken for 1 degree loss

      s += 1.

      optimizer.step()  # to update the weights with amount by which it will be updated
   
  
  print('epoch: '+str(epoch)+'loss: '+ str(train_loss/s))



"""## Testing the SAE"""



test_loss = 0

s = 0.

for id_user in range(nb_users):

  input = Variable(training_set[id_user]).unsqueeze(0)

  target = Variable(test_set[id_user]).unsqueeze(0)


    if torch.sum(target.data > 0) > 0:
  
    output = sae(input)
  
    target.require_grad = False
  
    output[target == 0] = 0
  
    loss = criterion(output, target)
  
    mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
  
    test_loss += np.sqrt(loss.data*mean_corrector)
  
    s += 1.


print('test loss: '+str(test_loss/s))