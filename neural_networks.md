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

