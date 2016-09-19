import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  dW = np.zeros_like(W)
  for i in xrange(num_train):
    x_i = X[:, i]
    scores = W.dot(x_i)
    scores -= np.max(scores)
    sum_exp_scores = np.sum(np.exp(scores))
    loss += np.log(sum_exp_scores)
    loss -= scores[y[i]]
    for j in xrange(num_classes):
      dW[j] +=  np.exp(scores[j]) * x_i/sum_exp_scores
    dW[y[i]] -= x_i


  loss /= num_train
  dW /= num_train


  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += 2 * reg * W

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[1]
  scores = W.dot(X)
  scores -= np.max(scores, axis=0)
  sum_exp_scores = np.sum(np.exp(scores), axis=0)
  loss = np.sum(np.log(sum_exp_scores))
  correct_scores = scores[y, range(scores.shape[1])]
  loss -= np.sum(correct_scores)

  dW = (np.exp(scores)/sum_exp_scores).dot(X.T)

  dW_yis_contribs = np.zeros((W.shape[0], X.shape[1]))
  dW_yis_contribs[y, range(X.shape[1])] = 1
  dW -= dW_yis_contribs.dot(X.T)

  loss /= num_train
  dW /= num_train


  loss += 0.5 * reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
