import numpy as np
from .data_loader import data_set
from .preprocessing import preprocess_data
from .modal1 import modal

def modal_testing(data, weight):
        x_train_scale, x_test_scale, y_train, y_test,means, std= data
        add_one = np.ones((x_test_scale.shape[0], 1))
        x_test = np.column_stack((x_test_scale, add_one))
        # print(x_test[0])
        prediction=(x_test@ weight)      # prediction
        prediction=np.array(prediction, dtype=float)
        # print(prediction[:10],"\n", y_test[:10])
        # print(prediction[:10]-y_test[:10])
        return prediction
def testing_error(data,prediction):
                x_train_scale, x_test_scale, y_train, y_test, means, std = data
                error= prediction- y_test
                MAE=np.sum(abs(error))/len(prediction)
                MSE= np.mean(error ** 2)
                RMSE = np.sqrt(MSE)
                MSE=round(MSE,2)
                MAE=round(MAE, 2)
                RMSE=round(RMSE,2)
                # print(MAE, RMSE, MSE)
                return MAE, RMSE, MSE


data=data_set()
set=preprocess_data(data)
weight=modal(set)
prediction=modal_testing(set, weight)
testing_error(set,prediction)
