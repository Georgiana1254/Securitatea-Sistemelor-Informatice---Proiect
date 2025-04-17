import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras._tf_keras.keras.optimizers import Adam
import tensorflow as tf



def load_and_normalize_csv():
    df1 = pd.read_csv("3.legitimate.csv", header=None, skiprows=1)
    df2 = pd.read_csv("4.phishing.csv", header=None, skiprows=1)

    df1.drop(columns=[0], inplace=True)
    df2.drop(columns=[0], inplace=True)

    df = pd.concat([df1, df2], ignore_index=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_df = np.array(scaler.fit_transform(df.values))
    df = pd.DataFrame(scaled_df)

    x = df.iloc[:, 1:17]
    x.to_csv("Normalizare.csv", header=None, index=False)

    file1 = pd.read_csv("Normalizare.csv", nrows=5000, header=None)
    file2 = pd.read_csv("Normalizare.csv", skiprows=5000, nrows=10000, header=None)
    file3 = pd.concat([file1, file2], ignore_index=True)

    return file1, file2, file3



def create_labels():
    file1, file2, _ = load_and_normalize_csv()

    num_phishing = len(file1)
    num_legitimate = len(file2)
    total_samples = num_phishing + num_legitimate

    labels = np.array(
        [1, 0] * num_phishing +
        [0, 1] * num_legitimate
    )

    return labels, total_samples



def build_lstm_model(input_dim, timesteps):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, input_shape=(timesteps, input_dim), return_sequences=True),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model



def train_model(model, X_train, y_train, X_test, y_test):
    optimizer = Adam(learning_rate=0.1)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
    return history



def evaluate_and_plot(model, history, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {test_loss}, Accuracy: {test_accuracy}")
    model.save('model_lstm.h5')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy - LSTM')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss - LSTM')
    plt.show()

    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_test_labels, y_pred_labels)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix - LSTM')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def main():
    timesteps = 1
    input_dim = 16

    file1, file2, file3 = load_and_normalize_csv()
    labels, _ = create_labels()

    data = file3.values.reshape(-1, 1, 16)
    labels = labels.reshape(-1, 2)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.90, random_state=30)

    model = build_lstm_model(input_dim, timesteps)
    history = train_model(model, X_train, y_train, X_test, y_test)
    evaluate_and_plot(model, history, X_test, y_test)


if __name__ == "__main__":
    main()
