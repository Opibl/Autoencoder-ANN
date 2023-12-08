# My Utility : auxiliars functions

import pandas as pd
import numpy as np

# load config param
def load_config():
    sae_config='cnf_sae.csv'
    soft_config='cnf_softmax.csv'

    # Open configs files and save in p_sae and p_sft
    with open(sae_config, 'r') as config_ae_csv, open(soft_config, 'r') as config_softmax_csv:
        p_sae = [int(i) if '.' not in i else float(i) for i in config_ae_csv if i.strip() != '']
        p_sft = [int(i) if '.' not in i else float(i) for i in config_softmax_csv if i.strip() != '']

    return p_sae, p_sft


# Initialize weights for SNN-SGDM
def iniWs(inshape, layer_node):
    W1 = iniW(layer_node, inshape)
    W2 = iniW(inshape, layer_node)
    W = list((W1, W2))

    V = []
    for i in range(len(W)):
        V.append(np.zeros(W[i].shape))

    return W, V


# Initialize weights for one-layer
def iniW(next, prev):
    r = np.sqrt(6/(next + prev))
    w = np.random.rand(next, prev)
    w = w*2*r-r
    return w


# Feed-forward of SNN
def forward_ae(X, W, Param):
    act_func = Param[1] # Encoder Activation

    A = []
    z = []
    Act = []

    # Data Input
    z.append(X)
    A.append(X)
    
    for i in range(len(W)):
      
        X = np.dot(W[i], X)
        z.append(X)
        if i == 0:
            X = act_function(X, act_func)

        A.append(X)

    Act.append(A)
    Act.append(z)

    return Act


# Activation function
def act_function(x, act=1):

    # Default Values
    a_ELU=1
    a_SELU=1.6732
    lambd=1.0507

    # Relu
    if act == 1:
        condition = x > 0
        return np.where(condition, x, np.zeros(x.shape))

    # LRelu
    if act == 2:
        condition = x >= 0
        return np.where(condition, x, x * 0.01)

    # ELU
    if act == 3:
        condition = x > 0
        return np.where(condition, x, a_ELU * np.expm1(x))

    # SELU
    if act == 4:
        condition = x > 0
        return lambd * np.where(condition, x, a_SELU * np.expm1(x))

    # Sigmoid
    if act == 5:
        return 1 / (1 + np.exp(-1*x))

    return x


# Derivatives of the activation funciton
def deriva_act(x, act=1):

    # Default Values
    a_ELU=1
    a_SELU=1.6732
    lambd=1.0507
    
    # Relu
    if act == 1:
        condition = x > 0
        return np.where(condition, np.ones(x.shape), np.zeros(x.shape))
   
    # LRelu
    if act == 2:
        condition = x > 0
        return np.where(condition, np.ones(x.shape), np.ones(x.shape) * 0.01)

    # ELU
    if act == 3:
        condition = x > 0
        return np.where(condition, np.ones(x.shape), a_ELU * np.exp(x))

    # SELU falta
    if act == 4:
        condition = x > 0
        return lambd * np.where(condition, np.ones(x.shape), a_SELU * np.exp(x))

    # Sigmoid
    if act == 5:
        # pasarle la sigmoid
        return np.multiply(act_function(x, act=5), (1 - act_function(x, act=5)))

    return x


# Calculate Pseudo-inverse
def pinv_ae(x, H, C):
    A = np.dot(H, H.T) + (1/C)
    A_inv = np.linalg.pinv(A)
    w2 = np.linalg.multi_dot([x, H.T, A_inv])

    return(w2)

 
#Feed-Backward of SNN
def gradW_ae(Act, W, Param):
    act_func = Param[1] # Encoder Activation

    L = len(Act[0])-1

    M = Param[3]

    e = Act[0][L] - Act[0][0]

    Cost = np.sum(np.sum(np.square(e), axis=0)/2)/M

    # Gradient Decoder
    delta = e
    gW_l = np.dot(delta, Act[0][L-1].T)/M
    gW = []
    gW.append(gW_l)

    # Gradient Encoder
    t1 = np.dot(W[1].T, delta)

    t2 = deriva_act(Act[1][1], act_func)

    t3 = Act[0][0].T

    gW_l = np.dot(np.multiply(t1, t2), t3)/M

    gW.append(gW_l)
    gW.reverse()
    return gW, Cost


# Update W and V
def updWV_RMSprop(W, V, gW, tasa):

    # Default Values
    eps = 1e-8
    beta = 0.9

    for i in range(len(W)):    
        
        V[i] = (beta * V[i]) + ((1-beta)*gW[i]**2)
        
        W[i] = W[i] -( (tasa /np.sqrt(V[i]+eps)) * gW[i])
    
    return W, V


def updWV_RMSprop2(W, V, gW, tasa):
    
    # Default Values
    eps = 1e-8
    beta = 0.9
     
    V = (beta * V) + ((1-beta)*gW**2)
    
    W= W -( (tasa /np.sqrt(V+eps)) * gW)
    
    return W, V


def gradW(Act, W, Param):
    
    act_function = Param[1] # Encoder Activation
    M = Param[3]
    L = len(Act[0])-1
    
    gW = []

    Cost = np.sum(np.sum(np.square(Act[0][L] - Act[0][0]), axis=0)/2)/M
    
    # Gradient to Output
    delta = np.multiply(Act[0][L] - Act[0][0], 1)
    gW_l = np.dot(delta, Act[0][L-1].T)/M

    gW.append(gW_l)

    # Gradient to hidden Layers
    for l in reversed(range(1,L)):
        
        t1 = np.dot(W[l].T, delta)
        t2 = deriva_act(Act[1][l], act_function)
        delta = np.multiply(t1, t2)
        t3 = Act[0][l-1].T

        gW_l = np.dot(delta, t3)/M
        gW.append(gW_l)

    gW.reverse()
    return gW, Cost


# Softmax's gradient
def gradW_softmax(x,y,a):
    M      = y.shape[1]
    Cost   = -(np.sum(np.sum(  np.multiply(y,np.log(a)) , axis=0)/2))/M
    
    gW     = -(np.dot(y-a,x.T))/M
    return gW, Cost
    

# Calculate Softmax
def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return (exp_z/np.sum(exp_z,axis=0,keepdims=True))


# Save weights and MSE  of the SNN
def save_w_dl(W,Ws,Cost):
    np.savez('wAEs.npz', W[0], W[1])                    # Save AEs weights
    np.savez('wSoftmax.npz', Ws)                        # Save Softmax weights
    
    df = pd.DataFrame( Cost )
    df.to_csv('costo.csv',index=False, header = False ) # Save Cost csv
    
    return

# -----------------------------------------------------------------------