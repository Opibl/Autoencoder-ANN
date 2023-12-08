import pandas as pd
import numpy as np
import utility as ut


# load data for testing
def load_data_tst():
    tst = np.load('test.npz') # Load test.npz
    return [tst[i] for i in tst.files]

# load weight of the DL in numpy format
def load_w_dl():
    ws_ae = np.load('wAEs.npz')                     # Load AEs weights
    ws_soft = np.load('wSoftmax.npz')               # Load Softmax weights
    ws = [ws_ae[i] for i in ws_ae.files]            # Save AEs weights in ws
    ws.extend([ws_soft[i] for i in ws_soft.files])  # Save Softmax weights in ws
    return ws

# Feed-forward of the DL
def forward_dl(X, W):

    for i in range(len(W)):
        X = np.dot(W[i], X)
        if i == len(W)-1:
            X = ut.softmax(X)
        else:
            X = ut.act_function(X, 2)  
            
    return X

# Measure
def metricas(y, z): 
    cm, cm_m = confusion_matrix(y, z)

    Fsc = []
    for i in range(len(cm_m)):
        TP = cm_m[i, 0, 0]
        FP = cm_m[i, 0, 1]
        FN = cm_m[i, 1, 0]
        TN = cm_m[i, 1, 1]

        Precision = TP / (TP + FP)                                  # Calculate Precission
        Recall = TP / (TP + FN)                                     # Calculate Recall
        Fsc.append((2 * Precision * Recall) / (Precision + Recall)) 

    Fsc.append(sum(Fsc)/len(Fsc)) 


    Fsc = np.asarray(Fsc)

    df_cm = pd.DataFrame(cm)
    df_cm.to_csv('cmatriz.csv', index=False, header = False)         # Save Confusion Matrix
    
    df_Fsc = pd.DataFrame(Fsc)
    df_Fsc.to_csv('fscores.csv', index=False, header = False)        # F-scores

    return cm, Fsc


def confusion_matrix(y, z):
    y,z = y.T,z.T
    m = y.shape[0]
    c = y.shape[1]
    
    y = np.argmax(y, axis=1)
    
    z = np.argmax(z, axis=1)
   
    cm = np.zeros((c, c))

    for i in range(m):
        cm[z[i], y[i]] += 1

    cm_m = np.zeros((cm.shape[0], 2, 2))                        # Confusion Matrix for classes

    for i in range(cm.shape[0]):
        cm_m[i, 0, 0] = cm[i, i]                                # True Positive
        cm_m[i, 0, 1] = np.sum(np.delete(cm[i, :], i, axis=0))  # False Positive
        cm_m[i, 1, 0] = np.sum(np.delete(cm[:, i], i, axis=0))  # False Negative
        cm_m[i, 1, 1] = np.sum(
            np.delete(np.delete(cm, i, axis=1), i, axis=0))     # True Negative

    return cm, cm_m

# Beginning ...
def main():
    print('Ejecutando Test')
    xv, yv = load_data_tst()
    W = load_w_dl()
    zv = forward_dl(xv, W)
    cm, Fsc = metricas(yv, zv)
    print(Fsc*100)
    print('Fsc-mean {:.5f}'.format(Fsc.mean()*100))


if __name__ == '__main__':
    main()
