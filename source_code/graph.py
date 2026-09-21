import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from source_code.data_loader import data_set
from source_code.preprocessing import preprocess_data
from source_code.modal1 import modal
from source_code.modal1_testing import testing_error, modal_testing


def reading_score(data, prediction):
    x_train_scale, x_test_scale, y_train, y_test, means, std = data
    plt.scatter(y_test, prediction)
    plt.xlabel("Actual Reading Score")
    plt.ylabel("Predicted Reading Score")
    plt.title("Actual vs Predicted Reading Score")
    plt.show()

def errorplot(data, prediction):
    x_train_scale, x_test_scale, y_train, y_test, means, std = data
    y_pre=prediction
    error=y_test-y_pre
    plt.scatter(y_pre,error )
    plt.axhline(0)
    plt.xlabel("Predicted Reading Score")
    plt.ylabel("Error in prediction")
    plt.title("Error Scatter Graph")

    plt.show()

def prediction_error(data, prediction):
    x_train_scale, x_test_scale, y_train, y_test, means, std = data
    y_pre=prediction
    error=y_test-y_pre
    errors = y_test - y_pre

    plt.hist(errors, bins=20)

    plt.xlabel("Prediction Error")
    plt.ylabel("Number of Students")
    plt.title("Distribution of Prediction Errors")
    plt.show()




data = data_set()
set_data = preprocess_data(data)
weight = modal(set_data)
prediction = modal_testing(set_data, weight)
error = testing_error(set_data, prediction)
reading_score(set_data, prediction)
errorplot(set_data, prediction)
prediction_error(set_data, prediction)