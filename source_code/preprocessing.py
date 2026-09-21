from .data_loader import data_set
import numpy as np


def preprocess_data(data):
    gender, race_encode,P_edu_level,lunch,test_pre_course,math_score,reading_score,write_score=data      # unpack data
    
    features = np.column_stack((
    gender,
    race_encode,
    P_edu_level,
    lunch,
    test_pre_course,
    math_score,
    write_score

    ))
    # print(features.shape)
 # features and reading_score variable
    y=np.array(reading_score)
    x=np.array(features)

   # cheaking shape of features(x)
    # print("Features shape:", x.shape)
    # print("Target shape:", y.shape)

    # print("Features dtype:", x.dtype)
    # print("Target dtype:", y.dtype)

    # print("First feature row:", x[0])
    # print("First target:", y[0])

# cheaking missing or invalid value

    # print("NaN in features:", np.isnan(x).sum())
    # print("NaN in target:", np.isnan(y).sum())

    
# SUFFELING DATA TO TRAN THE M0DAL
    np.random.seed(42)
    indices = np.random.permutation(len(x))
    x = x[indices]
    y = y[indices]


# spliting data in 8:2 ratio to train and test the modal
    x_train=x[:int(0.8*len(x))]
    x_test=x[int(0.8*len(x)):]

    y_train=y[:int(0.8*len(y))]
    y_test=y[int(0.8*len(y)):]

    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# scaling  train data
    means = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train_scale = (x_train - means) / std
    x_test_scale = (x_test - means) / std
    # print(x_train_scale.shape)
    # print(x_test_scale[0])

    return x_train_scale, x_test_scale, y_train, y_test, means, std


set_data=data_set()
preprocess_data(set_data)


