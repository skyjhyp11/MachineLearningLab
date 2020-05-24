from builtins import range
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_class = W.shape[1]
    num_train = X.shape[0]
    for i in xrange(num_train):
        Sj = X[i,:].dot(W) #(1,10)
        #Syi = X[i,:].dot(W[:,y[i]])
        Syi = Sj[y[i]]
        #exp_Syi = np.exp(Syi)
        exp_Sj = np.exp(Sj) #(1,10)
        #loss += -np.log((exp_Syi/np.sum(exp_Sj)))
        loss += (-Syi + np.log(np.sum(exp_Sj)))
        probability = exp_Sj/np.sum(exp_Sj) #(1,10)
        for j in xrange(num_class):
            if j != y[i]:
                dW[:,j] += probability[j] * X[i,:].T #(3073,1)
            else:
                dW[:,j] += (-1 + probability[j]) * X[i,:].T 
    dW /= num_train
    loss /= num_train
    loss += reg * np.sum(W*W)
    dW += reg * W
    #pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    
    scores = X.dot(W)  # 计算分数矩阵。 500x3073 * 3073x10 = 500x10
    exp_scores = np.exp(scores)  # e^scores.  500x10 
    sum_scores = np.sum(exp_scores, axis = 1)  # 各个类所对应的分数总和。 (500,)
    exp_scores /= sum_scores[:,np.newaxis]  # 标准化后的概率 500x10/=500x1
    loss_matrix = -np.log(exp_scores[range(num_train),y])  # 500,
    loss += np.sum(loss_matrix)
    exp_scores[range(num_train),y] -= 1  # 取正确标签处，减一
    dW += np.dot(X.T, exp_scores)  # 3073x500 * 500x10 = 3073x10

    loss /= num_train
    loss += reg * np.sum(W*W)
    dW /= num_train
    dW +=reg *W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def normalized(a):
    sum_scores = np.sum(a,axis=0)
    sum_scores = 1 / sum_scores
    result = a.T * sum_scores.T
    return result.T
