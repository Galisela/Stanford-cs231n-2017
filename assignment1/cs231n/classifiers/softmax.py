import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    f = X[i,:].dot(W)
    f -= np.max(f)
    p = np.exp(f) / np.sum(np.exp(f))
    
    loss += -np.log(p[y[i]])
    
    for k in range(num_classes):
        dW[:,k] += (p[k]-(y[i]==k))*X[i,:]
    
    
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /=  num_train 

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg*2*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  F = X.dot(W)                       #size (num_train,num_classes)
  F = (F.T - np.max(F,axis=1)).T
  P = (np.exp(F.T) / np.sum(np.exp(F),axis=1)).T
  
  # The contribution of each Xi to the loss is -log(exp(Wyi_dot_Xi)/sum_over_j(exp(Wj_dot_Xi))  IE the correct class item in line i of P
  loss = - np.sum(np.log(P[range(num_train),y]))
  
    
  # The contribution of each Xi to dW_k is P(i,k)*Xi if k=/=y[i] and (P(i,k)-1)*Xi if k=y[i] (i runs on samples and k on features)
  # if X.T_dot_P = M  (Xi are organized in collums) then M(:,k) = sum_over_i(P_ik*Xi) to get dW all that is left is to substruct 1
  # from P where k==y[i]
  P[range(num_train),y] -=1  
  dW =  X.T.dot(P)
                  
                  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /=  num_train 

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg*2*W
  
       

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

