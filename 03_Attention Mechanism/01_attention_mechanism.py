import numpy as np

#Define Query, Key, Value matrices
Q = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0]])
K = np.array([[0, 1, 0], [1, 0, 1], [1, 0, 0]])
V = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]])

#Compute attention scores
scores = np.dot(Q,K.T)

#Apply Softmax to get Attention Weights
def softmax(x):
    exp_x = np.exp(x-np.max(x,axis = -1, keepdims=True))
    return exp_x / exp_x.sum(axis = -1, keepdims=True)

attention_weights = softmax(scores)

#Compute Attention Output
attention_output = np.dot(attention_weights, V)

print(f"Attention Weights: {attention_weights}")
print(f"Attention Output: {attention_output}")