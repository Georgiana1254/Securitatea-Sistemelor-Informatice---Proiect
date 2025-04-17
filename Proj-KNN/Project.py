import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)
import joblib

def import_csv():
    df1 = pd.read_csv("3.legitimate.csv", header=None, skiprows=1)
    df2 = pd.read_csv("4.phishing.csv", header=None, skiprows=1)
    
    df1.drop(columns=[0], inplace=True)
    df2.drop(columns=[0], inplace=True)
    
    df = pd.concat([df1, df2], ignore_index=True)
    df = pd.DataFrame(df)
    
    features = df.iloc[:, 1:17]
    features.to_csv("Normalizare.csv", header=None, index=False)
    
    file1 = pd.read_csv("Normalizare.csv", nrows=5000, header=None)
    file2 = pd.read_csv("Normalizare.csv", skiprows=5000, nrows=10000, header=None)
    file3 = pd.concat([file1, file2], ignore_index=True)

    return file1, file2, file3


def create_labels():
    file1, file2, _ = import_csv()
    num_phishing = len(file1)
    num_legitimate = len(file2)
    
    etichete = np.array(
        [1] * num_phishing + 
        [0] * num_legitimate
    )
    return etichete


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title('Confusion Matrix - KNN')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def plot_learning_curve(X, y):
    train_sizes = [1, 500, 1000, 2000, 3000, 8000]
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=KNeighborsClassifier(n_neighbors=1),
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=5,
        scoring='accuracy'
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, label='Training Accuracy', marker='o')
    plt.plot(train_sizes, val_mean, label='Validation Accuracy', marker='s')
    plt.ylabel('Accuracy')
    plt.xlabel('Training Set Size')
    plt.title('Learning Curves for KNN Model')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(0.95, 1.01)
    plt.show()

def create_the_model():
    file1, file2, file3 = import_csv()
    etichete = create_labels()
    
    X = file3.values
    y = etichete

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10
    )

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)

    joblib.dump(knn, 'model_knn.pkl')

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    plot_confusion_matrix(y_test, y_pred)
    plot_learning_curve(X, y)


if __name__ == "__main__":
    import_csv()
    create_the_model()
