import numpy as np
from scipy.signal import welch
from sklearn import ensemble
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

user_acc_list = []
acc_list = []
for user_number in range(1,23):
    user_acc_list = []
    for training_num in range(3,16):
        y_val = []
        y_train = []
        features = []


        for i in range(0, training_num):  
            filename = f'../user_extracted_data/user{user_number:02}/user{user_number:02}_true_{i}.npy'  
            data = np.load(filename, allow_pickle=True)
            psd_list = []
            for k in range(0,32):
                psd = welch(data[k])
                psd_list.append(psd[1])
            features.append(np.mean(psd_list, axis=0))
        x_train = np.array(features)
       

        features = []
        for i in range(15, 20):  
            filename = f'../user_extracted_data/user{user_number:02}/user{user_number:02}_true_{i}.npy'  
            data = np.load(filename, allow_pickle=True)
            psd_list = []
            for k in range(0,32):
                psd = welch(data[k])
                psd_list.append(psd[1])
            features.append(np.mean(psd_list, axis=0))
            y_val.append(1)

        for i in range(0, 10):
            filename = f'../user_extracted_data/user{user_number:02}/user{user_number:02}_false_{i}.npy' 
            data = np.load(filename, allow_pickle=True)
            psd_list = []
            for k in range(0,32):
                psd = welch(data[k])
                psd_list.append(psd[1])
            features.append(np.mean(psd_list, axis=0))
            y_val.append(-1)

        x_val = np.array(features)
        y_val = np.array(y_val)
       
        # classifier = covariance.EllipticEnvelope()
        classifier = ensemble.IsolationForest()
        # classifier = svm.OneClassSVM(kernel='rbf')
        classifier.fit(x_train)
        preds=classifier.predict(x_val)
      
        accuracy = accuracy_score(y_val, preds)
        user_acc_list.append(accuracy)
        
    print(preds)
    acc_list.append(user_acc_list)
    print(f'Max accuracy: {max(user_acc_list)*100:.2f}%, No. of training samples: {np.argmax(user_acc_list)+2}')

acc_list = np.array(acc_list)
fig, ax = plt.subplots()
h,w = np.shape(acc_list)

for i in range(w):
        for j in range(h):
            val = acc_list[j, i]
            ax.text(i, j, f'{val*100:.2f}', va='center', ha='center')
ax.imshow(acc_list, cmap='YlGn')
x_labels = [str(i) for i in range(3,w+3)]
y_labels = [str(i) for i in range(1,h+1)]
print(len(x_labels))
print(len(y_labels))
ax.set_xticks(np.arange(len(x_labels)))
ax.set_yticks(np.arange(len(y_labels)))

ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)

fig.set_figwidth(w)
fig.set_figheight(h+1)
plt.ylabel('User Number')
plt.xlabel('No. of training samples')
plt.title(type(classifier))
plt.subplots(2,4)
for i in range(0,8):
    plt.subplot(2,4,i+1)
    plt.title(f'User {i+1:02}')
    plt.plot(np.arange(3,w+3),acc_list[i], '-b')
    plt.xlim(3, w+2)
    plt.ylim(0, 1)
    plt.grid(True)
plt.subplots(2,4)
for i in range(0,8):
    plt.subplot(2,4,i+1)
    plt.title(f'User {i+9:02}')
    plt.plot(np.arange(3,w+3),acc_list[i+8], '-b')
    plt.xlim(3, w+2)
    plt.ylim(0, 1)
    plt.grid(True)

plt.subplots(2,3)
for i in range(0,6):
    plt.subplot(2,3,i+1)
    plt.title(f'User {i+17:02}')
    plt.plot(np.arange(3,w+3),acc_list[i+16], '-b')
    plt.xlim(3, w+2)
    plt.ylim(0, 1)
    plt.grid(True)
plt.show()