import numpy as np

from sklearn.model_selection import train_test_split
from scipy.signal import welch
import numpy as np
from sklearn import svm, tree, neighbors, discriminant_analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def prepare_data(user_number):
    j=0
    y_train = []
    features = []
    for i in range(0, 20):  #Number of valid attempts
        filename = f'../user_extracted_data/user{user_number:02}/user{user_number:02}_true_{i}.npy'  # Valid attempt filename
        data = np.load(filename, allow_pickle=True)
        psd_list = []
        for k in range(0,32):   #For each channel
                psd = welch(data[k])
                psd_list.append(psd[1])
        features.append(np.mean(psd_list, axis=1))
        y_train.append(1)   #Add a True label
        if j<10:    #Invalid attempts are 10, half the valid
            filename = f'../user_extracted_data/user{user_number:02}/user{user_number:02}_false_{j}.npy'  # Invalid attempt files
            data = np.load(filename, allow_pickle=True)
            psd_list = []
            for k in range(0,32):
                psd = welch(data[k])
                psd_list.append(psd[1])
            features.append(np.mean(psd_list, axis=1))
            y_train.append(0)   #Add a False label
            j+=1
    return features, y_train
     
if __name__ == '__main__':
    # user_number = 1
    accuracy_list=[0]*7
    prediction_list = []
    y_test_list = []
    for user_number in range(1,23):
        features, y_train = prepare_data(user_number)

        # Turn lists into numpy arrays
        x_train = np.array(features)
        y_train = np.array(y_train)

        # Split training and validation data randomly
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=93)


        classifier = neighbors.KNeighborsClassifier(n_neighbors=3)
        classifier.fit(x_train, y_train)
        preds=classifier.predict(x_val)
        accuracy_list[0]+= accuracy_score(y_val, preds)
        for y in preds:
            prediction_list.append(y)
        for y in y_val:
            y_test_list.append(y)

        classifier = svm.SVC(kernel='linear')
        classifier.fit(x_train, y_train)
        preds=classifier.predict(x_val)
        accuracy_list[1]+= accuracy_score(y_val, preds)
       
        classifier = svm.SVC(kernel='poly')
        classifier.fit(x_train, y_train)
        preds=classifier.predict(x_val)
        accuracy_list[2]+= accuracy_score(y_val, preds)
        
       
        classifier = svm.SVC(kernel='sigmoid')
        classifier.fit(x_train, y_train)
        preds=classifier.predict(x_val)
        accuracy_list[3]+= accuracy_score(y_val, preds)
       
        classifier = svm.SVC(kernel='rbf')
        classifier.fit(x_train, y_train)
        preds=classifier.predict(x_val)
        accuracy_list[4]+= accuracy_score(y_val, preds)
       
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(x_train, y_train)
        preds=classifier.predict(x_val)
        accuracy_list[5]+= accuracy_score(y_val, preds)
       
        classifier = discriminant_analysis.LinearDiscriminantAnalysis()
        classifier.fit(x_train, y_train)
        preds=classifier.predict(x_val)
        accuracy_list[6]+= accuracy_score(y_val, preds)
        
        
        
        
        
    for accuracy in accuracy_list:
        print(f'{100*accuracy/22:.2f}%')
    result = confusion_matrix(y_test_list, prediction_list , normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=result, display_labels=[str(i) for i in range(0,2)])
    disp.plot(cmap='YlGn')
    plt.show()