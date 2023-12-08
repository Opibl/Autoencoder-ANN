# SNN's Training :

import pandas as pd
import numpy as np
import utility as ut

# gets Index for n-th miniBatch
def get_Idx_n_Batch(n, M):
    Idx = (n*M, (n*M)+M)
    return(Idx)


# Training miniBatch for softmax
def train_sft_batch(X, Y, W, V, Param):
    costo = []
    learning_rate = Param[1]
    M = Param[2]
    numBatch = np.int16(np.floor(X.shape[1]/M))

    for n in range(numBatch):
        Idx = get_Idx_n_Batch(n, M)
        xe, ye = X[:,slice(*Idx)], Y[:,slice(*Idx)]
        
        z = np.dot(W, xe)
        a = ut.softmax(z)
        
        gW, Cost = ut.gradW_softmax(xe, ye, a)
        
        W, V = ut.updWV_RMSprop2(W, V, gW, learning_rate)

        costo.append(Cost)

    return W, V, costo


# Softmax's training via SGD with Momentum
def train_softmax(X, Y, Param):    
    W = ut.iniW(Y.shape[0], X.shape[0])
    V = np.zeros(W.shape)

    Cost = []
    for Iter in range(1,Param[0]+1):
        idx   = np.random.permutation(X.shape[1])
        xe,ye = X[:,idx],Y[:,idx]   
        
        W, V, c = train_sft_batch(xe, ye, W, V, Param)

        Cost.append(np.mean(c))
    return W, Cost


# AE's Training with miniBatch
def train_ae_batch(X, W, v, Param):
    
    p_inverse       = Param[0]
    act_func    = Param[1]
    mini_batch_size = Param[3]
    learning_rate   = Param[4]

    numBatch = np.int16(np.floor(X.shape[1]/mini_batch_size))
    
    cost = []
    
    W[1] = ut.pinv_ae(X, ut.act_function(np.dot(W[0], X), act_func), p_inverse)  
    
    for n in range(numBatch):
        Idx = get_Idx_n_Batch(n, mini_batch_size)
        xe= X[:,slice(*Idx)]
        
        Act = ut.forward_ae(xe, W, Param)
        
        gW, Cost = ut.gradW_ae(Act, W, Param)
        
        W_1, v_1 = ut.updWV_RMSprop(W, v, gW,tasa = float(learning_rate)/100)
        
        W[0], v[0] = W_1[0], v_1[0]
        
        cost.append(Cost)
    
    return W, v, cost


# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(X, ae_layers, Param):
    
    W, v = ut.iniWs(X.shape[0], ae_layers)
    
    Cost = []
    for Iter in range(1,Param[2]+1):
        xe = X[:, np.random.permutation(X.shape[1])]  # Sort Random 
    
        W, v, c = train_ae_batch(xe, W, v, Param)
        
        Cost.append(np.mean(c))
    return W


# SAE's Training
def train_sae(X, Param):

    act_func = Param[1] # Encoder Activation

    W = []
    NumAe = Param[5:]
    for i in range(len(NumAe)):
        ae_layers = NumAe[i]
        w1 = train_ae(X, ae_layers, Param)[0]
        X = ut.act_function(np.dot(w1, X), act_func)
        W.append(w1)
        
    return W, X


# Load data to train the SNN
def load_data_trn():
    trn = np.load('train.npz') # Load train.npz
    return [trn[i] for i in trn.files]


# Beginning ...
def main():
    print('Cargando la configuracion AEs y Softmax')
    p_sae,p_sft = ut.load_config()           
    print('Cargando la Data de Train')
    xe,ye       = load_data_trn()   
    print('Ejecutando Entrenamiento AEs')
    W,Xr        = train_sae(xe,p_sae)   
    print('Ejecutando Entrenamiento Softmax')      
    Ws, cost    = train_softmax(Xr,ye,p_sft)
    print('Guardando los Pesos')
    ut.save_w_dl(W,Ws,cost)
    print('Trn.py finalizado :)')


if __name__ == '__main__':
    main()
