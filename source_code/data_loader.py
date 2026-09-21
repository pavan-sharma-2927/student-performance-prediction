import csv
import numpy as np

def data_set():

    # features
    gender=[]
    race=[]
    parental_edu=[]
    lunch=[]
    test_pre_course=[]
    math_score=[]
    write_score=[]

    # target
    reading_score=[]

    # reading CSV file
    with open("data/StudentsPerformance.csv", 'r') as new_file:
        csv_reader=csv.reader( new_file)
        next(csv_reader)
        for line in csv_reader:
        
            gender.append(line[0])
            race.append(line[1])
            parental_edu.append(line[2])
            lunch.append(line[3])
            test_pre_course.append(line[4])
            math_score.append(line[5])
            reading_score.append(line[6])  # reading score
            write_score.append(line[7])

            
    # creating numpy array of given data set
    gender=np.array(gender)
    race=np.array(race)
    parental_edu=np.array(parental_edu)
    lunch=np.array(lunch)
    test_pre_course=np.array(test_pre_course)
    math_score=np.array(math_score)
    write_score=np.array(write_score)
    reading_score=np.array(reading_score)

    # shape of given data
    # print(gender.shape, race.shape,parental_edu.shape, lunch.shape, test_pre_course.shape, math_score.shape,reading_score.shape, write_score.shape )

    # numerical conversion of gender (0 male / 1 female)
    gender=np.where(gender=="male",0,1)

    # race converstion if race = 1 else 0
    race_group=np.array(["group A", "group B", "group C", "group D", "group E"])
    race_encode=np.zeros((len(race), len(race_group)))
    # print(race_encode.shape)
    for i in range(len(race)):
        for j in range(len(race_group)):
            if race[i]==race_group[j]:
                race_encode[i,j]=1

    # print(race_encode)

    # parential_education converstion
    edu_level= np.array([
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ])
    P_edu_level=np.zeros((len(parental_edu), len(edu_level)))
    for i in range(len(parental_edu)):
        for j in range(len(edu_level)):
            if parental_edu[i]==edu_level[j]:
                P_edu_level[i,j]=1

    # print(P_edu_level)

    # lunch numerical data standerd=1 else 0
    lunch=np.where(lunch=="standard", 1, 0)


    # test completed=1 else 0
    test_pre_course=np.where(test_pre_course=="completed",1, 0)

        # numerical conversion of score columns
    math_score = math_score.astype(np.int64)
    reading_score = reading_score.astype(np.int64)
    write_score = write_score.astype(np.int64)


    return (
        gender,
        race_encode,
        P_edu_level,
        lunch,
        test_pre_course,
        math_score,
        reading_score,
        write_score
    )
data_set()





