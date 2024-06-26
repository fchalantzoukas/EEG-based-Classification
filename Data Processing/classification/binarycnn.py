import numpy as np
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten,  MaxPool2D, Dropout
from sklearn.model_selection import train_test_split
from scipy.signal import welch
from keras.optimizers import Adam
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
        features.append(np.vstack(psd_list)) 
        y_train.append(1)   #Add a True label
        if j<10:    #Invalid attempts are 10, half the valid
            filename = f'../user_extracted_data/user{user_number:02}/user{user_number:02}_false_{j}.npy'  # Invalid attempt files
            data = np.load(filename, allow_pickle=True)
            psd_list = []
            for k in range(0,32):
                psd = welch(data[k])
                psd_list.append(psd[1])
            features.append(np.vstack(psd_list))
            y_train.append(0)   #Add a False label
            j+=1
    return features, y_train

def build_cnn():
    model = Sequential()

    model.add(Input(shape=(32, 129, 1)))

    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=300))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    # print(model.summary())
    return model

def plot_metrics(epochs):
    fig,ax = plt.subplots()
    x=np.linspace(1,epochs,epochs)
    ax.plot(x,history.history['accuracy'], '-b', label = 'Accuracy')
    ax.plot(x,history.history['val_accuracy'],'-g', label = 'Validation Accuracy')
    ax.plot(x,history.history['loss'], '-r', label = 'Loss')
    ax.plot(x,history.history['val_loss'], '-m', label = 'Validation Loss')
    plt.xlim(0,epochs)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
     
if __name__ == '__main__':
    # user_number = 1
    accuracy_list=[]
    prediction_list = []
    y_test_list = []
    for user_number in range(1,23):
        features, y_train = prepare_data(user_number)

        # Turn lists into numpy arrays
        x_train = np.array(features)
        y_train = np.array(y_train)

        # Split training and validation data randomly
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=93)


        epochs=100
        model = build_cnn()
        history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), verbose=0)
        predictions = model.predict(x_val)
        predictions = np.round(predictions)
        for y in predictions:
            prediction_list.append(y)
        for y in y_val:
            y_test_list.append(y)
        accuracy_list.append(model.evaluate(x_val,y_val)[1])

        ## For single model enable the following
        # model.save('binary_cnn_classifier.keras')
        # predictions = model.predict(x_val)
        # predictions = np.round(predictions)
        # predictions = predictions.reshape(1, -1)
        # print(y_val)
        # print(predictions)
        # plot_metrics(epochs)
    accuracy = sum(accuracy_list)/len(accuracy_list)
    print(f'{100*accuracy:.2f}%')
    result = confusion_matrix(y_test_list, prediction_list , normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=result, display_labels=[str(i) for i in range(0,2)])
    disp.plot(cmap='YlGn')
    plt.show()