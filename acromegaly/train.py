from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import metrics
import numpy as np
import pandas as pd

kf  = KFold(n_splits=5,shuffle=True,random_state=20)

def train(model, data, label):
    accuracy_array = np.empty((0,4))
    sensitivity_array = np.empty((0,4))
    specificity_array = np.empty((0,4))

    for train_i, test_i in kf.split(data):

        x_train, x_test = data[train_i], data[test_i]
        y_train, y_test = label[train_i], label[test_i]
        model = model
        model.fit(x_train, y_train) 
        #create confusion matrix
        c_matrix = metrics.confusion_matrix(y_test, model.predict(x_test))  
        accuracy = accuracy_score(y_test, model.predict(x_test))
        sensitivity = c_matrix[0,0]/(c_matrix[0,0]+c_matrix[0,1])
        specificity = c_matrix[1,1]/(c_matrix[1,0]+c_matrix[1,1])   
        print(c_matrix)
        print('Accuracy : ', accuracy)
        print('Specificity : ', specificity)
        print('Sensitivity : ', sensitivity )

        accuracy_array = np.append(accuracy_array,accuracy)
        sensitivity_array = np.append(sensitivity_array, sensitivity)
        specificity_array = np.append(specificity_array, specificity)

    # Compute the values
    accuracy_mean = round(np.mean(accuracy_array), 2)
    accuracy_std = round(np.std(accuracy_array), 2)
    sensitivity_mean = round(np.mean(sensitivity_array), 2)
    sensitivity_std = round(np.std(sensitivity_array), 2)
    specificity_mean = round(np.mean(specificity_array), 2)
    specificity_std = round(np.std(specificity_array), 2)

    # Create a dictionary with your data
    data = {
        'Index': ['Accuracy', 'Sensitivity', 'Specificity'],
        'Mean': [accuracy_mean, sensitivity_mean, specificity_mean],
        'Std Deviation': [accuracy_std, sensitivity_std, specificity_std]
    }
    df = pd.DataFrame(data)
    print(df.head(5))
    df.to_excel(f"statistic/{model}.xlsx", engine="openpyxl")
