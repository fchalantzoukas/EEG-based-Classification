import numpy as np
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten,  MaxPool2D, SpatialDropout2D
from scipy.signal import welch
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def prepare_train_data():
    y_train = []
    features = []
    for user_number in range(1, 23):  
            for i in range(0,20):
                filename = f'../user_extracted_data/user{user_number:02}/user{user_number:02}_true_{i}.npy'  
                data = np.load(filename, allow_pickle=True)
                psd_list = []
                for k in range(0,32):
                    psd = welch(data[k])
                    psd_list.append(psd[1])
                features.append(np.vstack(psd_list))
                label = [0]*22
                label[user_number-1] = 1
                y_train.append(label)
    x_train = np.array(features)
    y_train = np.array(y_train)
    return x_train, y_train

def prepare_test_data():
    features = []
    y_val = []
    for user_number in range(1, 23):  
            k = user_number + 1
            for i in range(0,10):
                if k==23: k = 12
                if k==12 and user_number<12: k=1
                filename = f'../user_extracted_data/user{user_number:02}/user{user_number:02}_false_{i}.npy'  
                data = np.load(filename, allow_pickle=True)
                psd_list = []
                for channel in range(0,32):
                    psd = welch(data[channel])
                    psd_list.append(psd[1])
                features.append(np.vstack(psd_list))
                label = [0]*22
                label[k-1] = 1
                y_val.append(label)
                k+=1
    x_val = np.array(features)
    y_val = np.array(y_val)
    return x_val, y_val

def build_cnn():
    model = Sequential()

    model.add(Input(shape=(32, 129, 1)))

    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=2))
    model.add(SpatialDropout2D(0.5))
    model.add(Flatten())
    model.add(Dense(units=300))

    model.add(Dense(22, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def plot_metrics(epochs, predictions, y_val):
    fig,ax = plt.subplots()
    x=np.linspace(1,epochs,epochs)
    ax.plot(x,history.history['accuracy'], '-b', label = 'Accuracy')
    ax.plot(x,history.history['val_accuracy'],'-g', label = 'Validation Accuracy')
    ax.plot(x,history.history['loss'], '-r', label = 'Loss')
    ax.plot(x,history.history['val_loss'], '-m', label = 'Validation Loss')
    plt.xlim(0,epochs)
    plt.ylim(0,1)
    plt.legend(loc='best')
    plt.grid(True)
    y_prediction = np.argmax (predictions, axis = 1)
    y_test=np.argmax(y_val, axis=1)
    result = confusion_matrix(y_test, y_prediction , normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=result, display_labels=[str(i) for i in range(1,23)])
    disp.plot(cmap='YlGn')
    plt.show()
     
if __name__ == '__main__':
    
    x_train, y_train = prepare_train_data()
    x_val, y_val = prepare_test_data()

    epochs=100
    model = build_cnn()
    history = model.fit(x_train, y_train, batch_size=16, epochs=epochs, validation_data=(x_val, y_val), verbose=1)

    
    model.save('multiclass_cnn.keras')
    predictions = model.predict(x_val)
    predictions = np.round(predictions)
    print(f'{model.evaluate(x_val,y_val)[1]*100:.2f}%')
    plot_metrics(epochs, predictions, y_val)