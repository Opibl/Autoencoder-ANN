import pandas as pd
import numpy as np
import utility as ut


# Save Data
def save_data(Data, Label):
    np.savez('train.npz', Data[0], Label[0]) # Save data Train 
    np.savez('test.npz', Data[1], Label[1])  # Save data Test 
    return 


# normalize data
def data_norm(x):

    # Default Values
    a = 0.01
    b = 0.99

    max = np.max(x)                                         # Take Maximum Value
    min = np.min(x)                                         # Take Minimum Value

    data_norm = (((x-min)/(max-min))*(b-a))+a               # Calculate Normalized Data
    return data_norm


# Binary Label
def binary_label(classes):
    num_class = np.max(classes)                             # Calculation of number of classes
    classes = classes -1

    binary_label = np.zeros( (classes.shape[0],num_class) )
    binary_label[np.arange(0,len(classes)),classes] = 1     # Save Label of classes
    return binary_label


# Load data
def load_data_csv():
    DATA_csv = ['train.csv','test.csv']

    data = []
    label = []

    # Read Data Train and Test
    for archive in DATA_csv:
        data_class = np.genfromtxt(archive, delimiter=',')
        norm = data_norm(data_class[:,:-1].T)
        data.append(norm)                                              # Save Data 
        label.append(binary_label(data_class[:,-1].astype(int)).T )    # Save Labels
        
    return data,label


# Beginning ...
def main():
    print("Cargando archivos Train y Test")	
    Data, Label = load_data_csv()
    print("Guardando archivos train.npz y test.npz")
    save_data(Data, Label)
    print("Prep.py finalizado :)")
    

if __name__ == '__main__':
    main()
