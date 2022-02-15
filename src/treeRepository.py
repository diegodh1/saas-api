import base64

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from joblib import dump, load
from sklearn.metrics import plot_confusion_matrix
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
import math
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from graphviz import Source

file = pd.read_csv('https://drive.google.com/uc?export=download&id=1VZ8alKJLqTXG9_Td-LumwxlmsjkaHScq', header=None)

file = file.iloc[1:]
file.columns = [
    'age',
    'sex',
    'cp',
    'trestbps',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal',
    'target'

]

# we need to change the type of data in the columns to be able to make the graphs
file['age'] = file['age'].astype('float')
file['sex'] = file['sex'].astype('category')
file['cp'] = file['cp'].astype('category')
file['trestbps'] = file['trestbps'].astype('float')
file['chol'] = file['chol'].astype('float')
file['fbs'] = file['fbs'].astype('category')
file['restecg'] = file['restecg'].astype('category')
file['thalach'] = file['thalach'].astype('float')
file['exang'] = file['exang'].astype('category')
file['oldpeak'] = file['oldpeak'].astype('float')
file['slope'] = file['slope'].astype('category')
file['ca'] = file['ca'].astype('category')
file['thal'] = file['thal'].astype('category')
file['target'] = file['target'].astype('category')

data = {'age': file['age'], 'sex': file['sex'], 'cp': file['cp'], 'trestbps': file['trestbps'],
        'chol': file['chol'], 'fbs': file['fbs'], 'restecg': file['restecg'],
        'thalach': file['thalach'], 'exang': file['exang'], 'oldpeak': file['oldpeak'],
        'slope': file['slope'], 'ca': file['ca'], 'thal': file['thal'], 'target': file['target']}


# Print the score on the test data

def training_data(data):
    # delete duplicates
    data['age'].unique()
    data['sex'].unique()
    data['cp'].unique()
    data['trestbps'].unique()
    data['chol'].unique()
    data['fbs'].unique()
    data['restecg'].unique()
    data['thalach'].unique()
    data['exang'].unique()
    data['oldpeak'].unique()
    data['slope'].unique()
    data['ca'].unique()
    data['thal'].unique()
    data['target'].unique()
    # preparing the data
    dataTemp = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'thal', 'ca'])
    dataTemp.head()
    # training the model
    # target values
    y = dataTemp.target.values
    # all the data without the target column
    x = dataTemp.drop("target", axis=1)
    # split the sets
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, stratify=y, random_state=52)
    return [trainX, trainY, testX, testY]


def training_model(trainX, trainY):
    # create the classifier
    model_tree = tree.DecisionTreeClassifier(criterion="gini", max_depth=4)
    model_tree = model_tree.fit(trainX, trainY)
    return model_tree


def test_model(model_tree, testX, testY):
    score = model_tree.score(testX, testY)
    prediction = model_tree.predict(testX)
    result = []
    for res in prediction:
        result.append(res)
    return {'score': score, 'prediction': result}



def export_model():
    fileTemp = pd.DataFrame(file)
    model_data = training_data(fileTemp)
    model_tree = training_model(model_data[0], model_data[1])
    result = test_model(model_tree, model_data[2], model_data[3])
    dump(model_tree, 'filename.joblib')
    result['file'] = 'filename.joblib'
    return result

def show_model_information():
    fileTemp = pd.DataFrame(file)
    model_data = training_data(fileTemp)
    model_tree = load_model()
    result = test_model(model_tree, model_data[2], model_data[3])
    class_names = ["1", "2", "3", "4"]
    features_heart_labels = model_data[0].columns
    graph = Source(tree.export_graphviz(model_tree, out_file=None,
                                        feature_names=features_heart_labels,
                                        class_names=class_names, filled=True))
    graph.render('decision-tree')

    with open("decision-tree.pdf", "rb") as pdf_file:
        result['tree'] = base64.b64encode(pdf_file.read()).decode('utf-8')
    return result


def load_model():
    model_tree = load('filename.joblib')
    return model_tree


def predict_model(input_data):
    model_tree = load_model()
    new_data_frame = pd.DataFrame(
        {'age': input_data['age'],
         'sex': input_data['sex'],
         'trestbps': input_data['trestbps'],
         'chol': input_data['chol'],
         'fbs': input_data['fbs'],
         'thalach': input_data['thalach'],
         'exang': input_data['exang'],
         'oldpeak': input_data['oldpeak'],
         'cp_0': input_data['cp_0'],
         'cp_1': input_data['cp_1'],
         'cp_2': input_data['cp_2'],
         'cp_3': input_data['cp_3'],
         'restecg_0': input_data['restecg_0'],
         'restecg_1': input_data['restecg_1'],
         'restecg_2': input_data['restecg_2'],
         'slope_0': input_data['slope_0'],
         'slope_1': input_data['slope_1'],
         'slope_2': input_data['slope_2'],
         'thal_0': input_data['thal_0'],
         'thal_1': input_data['thal_1'],
         'thal_2': input_data['thal_2'],
         'thal_3': input_data['thal_3'],
         'ca_0': input_data['ca_0'],
         'ca_1': input_data['ca_1'],
         'ca_2': input_data['ca_2'],
         'ca_3': input_data['ca_3'],
         'ca_4': input_data['ca_4'],
         'target': input_data['target']}, index=[0])
    predict_data = prepare_predict_data(new_data_frame)
    return test_model(model_tree, predict_data[0], predict_data[1])


def prepare_predict_data(data):
    # preparing the data
    y = data.target.values
    print(y)
    # all the data without the target column
    x = data.drop("target", axis=1)
    return [x, y]

