import numpy as np
from .data_loader import data_set
from .preprocessing import preprocess_data


def modal(data):
    x_train_scale, x_test_scale, y_train, y_test,means,std = data


# adding intercept 1
    add_one=np.ones((x_train_scale.shape[0],1))
    intercept = np.column_stack((x_train_scale, add_one))
    # print(intercept.shape)
    XTX=intercept.T@intercept
    XTY=intercept.T@y_train
    weight = np.linalg.solve(XTX,XTY)
    # print(weight)
    return weight

    # XTY=np.transpose(x_train_scale)@y_train
    # XTX=np.transpose(x_train_scale)@x_train_scale
    # XW=y or w=XTX/XTY
 
    
    

data=data_set()
set = preprocess_data(data)
modal(set)
