import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data'
column_names = ['Class', 'T3', 'T4', 'TSH']
data = pd.read_csv(url, names=column_names, delim_whitespace=True)

class_map = {1: 'Hyperthyroid', 2: 'Hypothyroid', 3: 'Normal'}
data['Class'] = data['Class'].map(class_map)

imputer = SimpleImputer(strategy='mean')
data[['T3', 'T4', 'TSH']] = imputer.fit_transform(data[['T3', 'T4', 'TSH']])

discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
data[['T3', 'T4', 'TSH']] = discretizer.fit_transform(data[['T3', 'T4', 'TSH']])

X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Class']), data['Class'], test_size=0.3, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1)

model = BayesianNetwork([('T3', 'Class'), ('T4', 'Class'), ('TSH', 'Class')])

model.fit(train_data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)

def predict(model, X_test):
    y_pred = []
    for _, sample in X_test.iterrows():
        query = model.map_query(variables=['Class'], evidence=sample.to_dict())
        y_pred.append(query['Class'])
    return y_pred

y_pred = predict(inference, X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
