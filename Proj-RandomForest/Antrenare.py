import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    ConfusionMatrixDisplay
)


def import_csv():
    df1 = pd.read_csv("3.legitimate.csv", header=None, skiprows=1)
    df2 = pd.read_csv("4.phishing.csv", header=None, skiprows=1)

    df1.drop(columns=[0], inplace=True)
    df2.drop(columns=[0], inplace=True)

    df = pd.concat([df1, df2], ignore_index=True)

    x = df.iloc[:, 1:17]
    x.to_csv("Normalizare.csv", header=None, index=False)

    file = pd.read_csv("Normalizare.csv", nrows=5000, header=None)
    file2 = pd.read_csv("Normalizare.csv", skiprows=5000, nrows=10000, header=None)
    file3 = pd.concat([file, file2], ignore_index=True)

    return file, file2, file3



def create_labels():
    file1, file2, file3 = import_csv()

    num_samples_phishing = len(file1)
    num_samples_legitimate = len(file2)
    num_samples = num_samples_phishing + num_samples_legitimate

    etichete = np.array(
        [1] * num_samples_phishing +
        [0] * num_samples_legitimate
    )

    return etichete, num_samples


def create_the_model():
    file1, file2, file3 = import_csv()
    etichete, num_samples = create_labels()
    file = file3.values

    X_train, X_test, y_train, y_test = train_test_split(
        file, etichete,
        test_size=0.90,
        random_state=30
    )

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=30)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    joblib.dump(rf_classifier, 'model_rf.pkl')

    print(f"\nAccuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_rep)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('ConfusionMatrix10-90.pdf')
    plt.show()

    train_sizes = [1, 500, 1000, 2000, 3000, 8000]
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=RandomForestClassifier(),
        X=file,
        y=etichete,
        train_sizes=train_sizes,
        cv=5,
        scoring='accuracy'
    )

    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores_mean, label='Training Accuracy', marker='o')
    plt.plot(train_sizes, validation_scores_mean, label='Validation Accuracy', marker='s')
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Training Set Size', fontsize=14)
    plt.title('Learning Curves for Random Forest Model', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('10-90.pdf')
    plt.show()


if __name__ == "__main__":
    import_csv()
    create_the_model()
