# neural_networks
SVM loss function--judge whether we hava a good prediction

(wroing score-right score+loss value, 0)max

add regularizer penalty term into loss function to judge the parameters: L2: using square

using SVM you will get a score; using softmax you will get a probability--expectation

then normalize: using sigmoid function

score--exp--normalize--using the wrong category number--use log for loss function

softmax: sum=1

compare SVM and softmax:

SVM cannot make good, loss function would be 0

softmax: loss function won't be 0

BP: backpropogation  +gradient

find learning rate---step size*grad

BP:loss to parameters    find relationship between w and L

differenciate chain rule: find influence of variables to results   --gradients

activation function: sigmoid function: might cause gradient disapprear in some numbers when we do diffrenciate in multilayer

choose ReLU function instead

multi hidden neurons will get a better result ---but!!

overfitting problem! 

use regularized penalty λ

data: zero-centered---np.mean(x,axis=0)   normalized data--np.std(X,axis=0)  

intialize weights W: stochastic (gradient methods)
 
B=constant

Drop-out

```
import numpy as np
import matplotlib.pyplot as plt
```
set the plt parameters
```
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'
```
make random reproduce generation
```
np.random.seed(0)
```
get a matrix by 0, the input of np.zeros()should be one element, by defualt, type is float
```
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D))
y = np.zeros(N*K, dtype='uint8')
```
get a list of number randomly
```

for j in xrange(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
```
show it
```
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.show()
```

