import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data'
headers = ['Diagnosis', 'T3', 'T4', 'TSH']
df = pd.read_csv(data_url, names=headers, delim_whitespace=True)

label_mapping = {1: 'Hyperthyroid', 2: 'Hypothyroid', 3: 'Normal'}
df['Diagnosis'] = df['Diagnosis'].map(label_mapping)

df.fillna('Missing', inplace=True)

features_train, features_test, target_train, target_test = train_test_split(df.drop(columns=['Diagnosis']), df['Diagnosis'], test_size=0.3, random_state=42)
features_train['Diagnosis'] = target_train

bayes_net = BayesianNetwork([('T3', 'Diagnosis'), ('T4', 'Diagnosis'), ('TSH', 'Diagnosis')])
bayes_net.fit(features_train, estimator=MaximumLikelihoodEstimator)

infer = VariableElimination(bayes_net)

def make_predictions(model, test_features):
    predictions = []
    for _, instance in test_features.iterrows():
        evidence = instance.to_dict()
        try:
            result = infer.map_query(variables=['Diagnosis'], evidence=evidence)
            predictions.append(result['Diagnosis'])
        except KeyError as error:
            print(f"KeyError: {error} - Value may be missing in the CPD. Skipping instance.")
            predictions.append('Unknown')
    return predictions

predicted_labels = make_predictions(infer, features_test)
accuracy = accuracy_score(target_test, predicted_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')
