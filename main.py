import json
from ml_code.data_load import load_data
from ml_code.pre_processing import preprocess_data
from ml_code.models import ModelFactory
from ml_code.train import train_and_evaluate
from ml_code.metrics import print_metrics
from unit_test import TestDataLoader 



with open('config.json') as config_file:
    config = json.load(config_file)

data = load_data("D:/UMD Grad School/ENPM611/LabSession/data.csv")
print(data.head())


X_train, X_test, Y_train, Y_test = preprocess_data(data)

model = ModelFactory.get_model(config["model_type"])

accuracy, cm, y_test, y_prob = train_and_evaluate(model, X_train, X_test, Y_train, Y_test)

print_metrics(accuracy, cm, y_test, y_prob)
