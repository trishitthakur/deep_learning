'''from deep learning A-Z course
   coded by trishit nath thakur'''


"""### Importing libraries"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""## Importing dataset"""


dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values


"""## Feature Scaling"""


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) # feature range to scale data between the values (0,1)
X = sc.fit_transform(X)


"""##Training SOM"""


from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5) 

# dimension of grid 10 * 10, input_len is number of features in X, sigma is radius of different neighborhoods in the grid, lr is the convergence rate

som.random_weights_init(X)  # in minisom this function initilises the weights
som.train_random(data = X, num_iteration = 100) 

"""##Visualizing results"""


from pylab import bone, pcolor, colorbar, plot, show
bone() # initilize the window for map
pcolor(som.distance_map().T)  #  all values of mean inter neuron distance for all winning node of SOM
colorbar() # to show legend for the colors and find potential frauds by outliers
markers = ['o', 's'] # circle and square markers used
colors = ['r', 'g'] # red and green color


for i, x in enumerate(X):
    w = som.winner(x)  # winning node
    plot(w[0] + 0.5, # cordinate of marker at center of winning node
         w[1] + 0.5,
         markers[y[i]], # y[i] 1 or 0 according to if customer got approval(1) or not(0)
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

"""## Finding frauds"""


mappings = som.win_map(X) # dictionary of all mappings from winning node to the customers
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0) # getting whole list of cheaters 
frauds = sc.inverse_transform(frauds)

"""##Printing Fraunch Clients"""


print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))