import numpy as np
from source_code.data_loader import data_set
from source_code.preprocessing import preprocess_data
from source_code.modal1 import modal
from source_code.modal1_testing import testing_error, modal_testing



def get_user_input():
    print("========================================================================")
    print("||             STUDENT PERFORMANCE PREDICTION                         ||"  )
    print("|| Predicting Reading Score on the basis of given features            ||")
    print("========================================================================")

    gender = input("            Enter gender\n (male/female): ").lower()

    race = input("          Enter race/ethnicity\n (group A/B/C/D/E): ").upper()

    parental_edu = input("          Enter parental education \n(some high school/high school/some college/associate's degree/bachelor's degree/master's degree): ").lower()

    lunch = input("         Enter lunch \n(standard/free/reduced): ").lower()

    test_preparation = input("          Enter test preparation \n(completed/none): ").lower()

    math_score = get_valid_score("\nEnter math score (0-100): ")

    writing_score = get_valid_score("Enter writing score (0-100): ")

    return gender,race,parental_edu,lunch,test_preparation,math_score,writing_score

def get_valid_score(prompt):
    while True:
        try:
            value = float(input(prompt))
        except ValueError:
            print("Invalid input. Enter a number between 0 and 100.")
            continue
        if 0 <= value <= 100:
            return value
        print("Invalid score. Enter a value between 0 and 100.")

def encoding_user_input(data, set):

    gender,race,parental_edu,lunch,test_preparation,math_score,writing_score=data
    x_train_scale, x_test_scale, y_train, y_test, means, std=set
        # Encode gender

    if gender == "female":
        gender_encoded = 1
    elif gender == "male":
        gender_encoded = 0
    else:
        print("Invalid gender")
        exit()

    # One-hot encode race

    race_encoded = [0, 0, 0, 0, 0]

    race_index = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4
    }

    if race in race_index:
        race_encoded[race_index[race]] = 1
    else:
        print("Invalid race")
        exit()

    # One-hot encode parental education

    education = [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]

    education_encoded = [0] * len(education)

    if parental_edu in education:
        education_encoded[education.index(parental_edu)] = 1
    else:
        print("Invalid parental education")
        exit()

    # Encode lunch
    
    if lunch == "standard":
        lunch_encoded = 1
    elif lunch == "free/reduced" or lunch == "reduced" or lunch == "free":
        lunch_encoded = 0
    else:
        print("Invalid lunch")
        exit()

    # Encode test preparation
    
    if test_preparation == "completed":
        test_encoded = 1
    elif test_preparation == "none":
        test_encoded = 0
    else:
        print("Invalid test preparation")
        exit()

    # Create feature vector
    
    X = np.array(
        [
            gender_encoded,
            *race_encoded,
            *education_encoded,
            lunch_encoded,
            test_encoded,
            math_score,
            writing_score
        ],dtype=float)

    x_scale= (X - means) / std
    x_scale=np.append(x_scale,1)       
    # print(x_scale)
    return x_scale

def predict_reading_score(features, weight, error):
    # unboxing data
    x_scale=features
    w=weight
    MAE, RMSE, MSE=error
    # print(X,w,error)

    # reading score prediction
    reading_score=w@x_scale
    reading_score=np.clip(reading_score,0,100)

    print("========================================================================")
    print("||                        PREDICTION RESULT                           ||")
    print("========================================================================")
    print(f"                reading score : {round(reading_score,2)}                  \n")
    print(f"         Error can be generate in prediction   \n                ")
    print(f"            MAE Error: {MAE}\n            RMSE Error: {RMSE}\n            MSE Error: {MSE} ")
    print("========================================================================")






# loading data
data=data_set()
set=preprocess_data(data)
weight=modal(set)
prediction=modal_testing(set, weight)
error=testing_error(set,prediction) 

# main.py call
user_input=get_user_input()
features=encoding_user_input(user_input, set)
predict_score=predict_reading_score(features, weight, error)

 

