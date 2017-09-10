import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # The contribution of this part to the derivatives
        dW[:,j]+= X[i]
        dW[:,y[i]]+= -X[i]
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /=  num_train 

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg*2*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  num_classes = W.shape[1]  

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W) # size (num_train,num_classes)
  correct_class_scores = scores[range(num_train),y] 
  margins = (scores.T - correct_class_scores).T + 1   #All margines including negative and j=yi
  margins[range(num_train),y] = 0  # remove j=yi, can also just substruct num_train from loss at the end but I need this for dW  
  relevant_margins_bool = margins>0     
  margins = margins*relevant_margins_bool
                                                          
  loss = np.sum(margins)     
  loss /= num_train
  loss += reg * np.sum(W * W)  
                                                          


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  # Remember: margins[j,i] = max(0, Xj_dot_Wi - Xj_dot_Wyi+1) (yj=/=i)
  # each collumn i in margins contributes a sum of positive Xj_dot_Wi to the loss (for each collumn i, j runs on all Xs in which yj=/=i
  # AND in which the margins meet the proper criteria) 
  # When calculating the positive contribution to dW, for each collumn i, the derivative is the sum of the relevant Xj's. 
  
  # Notice, if we have a matrix X (features,num_train), multiplied by a matrix M (num_train,classes) a 
  # value of 1 in M[1,2] means adding the first collumn in X to the second collumn in the result. The algebra is a superposition of that  
  
  pos_dW = X.T.dot(relevant_margins_bool)  
  
  neg_margin_contributions = relevant_margins_bool.sum(axis=1) #how many margins with -Xi_dot_Wyi contribution are in each row i
  # Now each negative contribution of Xi to dW should go to the currect collumn (collumn yi). Same method as with the positive part
  neg_cont_to_coll = np.zeros(relevant_margins_bool.shape)   # size=(num_train,num_classes)
  neg_cont_to_coll[range(num_train),y] = neg_margin_contributions 
  neg_dW = X.T.dot(neg_cont_to_coll) #Each contribution went to the right collumn 
    
    

  dW = pos_dW -  neg_dW
  dW /=  num_train
  dW += reg*2*W   #the contribution of the regularization to dW
  
    

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
