import pandas as pd
import tensorflow as tf
import numpy as np
import random as rnd

TRAIN_FILEPATH = "2015_data.csv"
TEST_FILEPATH = "2014_data.csv"

ESTIMATOR = None

def readFile(filename):
    cols = ["manner_of_death", "resident_status", "sex", "marital_status", "detail_age", "race", ]
    file = pd.read_csv(filename, nrows=50000, usecols=cols)
    file.dropna(axis=0, how="any", inplace=True)
    original = file["manner_of_death"]
    cats = file["manner_of_death"].astype("category").cat.codes
    print(list(zip(cats.unique(), original.unique())))
    file["manner_of_death"] = cats
    file["manner_of_death"] = file["manner_of_death"].astype(np.int32)
    file["sex"] = file["sex"].astype("category").cat.codes
    file["marital_status"] = file["marital_status"].astype("category").cat.codes
    file["sex"] = file["sex"].astype(np.int32)
    file["marital_status"] = file["marital_status"].astype(np.int32)
    return file


def predict(trainFile, testFile, hidden, trainLabel, testLabel):
    resident_status = tf.contrib.layers.sparse_column_with_integerized_feature("resident_status", 4)
    sex = tf.contrib.layers.sparse_column_with_integerized_feature("sex", 2)
    detail_age = tf.feature_column.numeric_column("detail_age")
    marital_status = tf.contrib.layers.sparse_column_with_integerized_feature("marital_status", 5)
    race = tf.contrib.layers.sparse_column_with_integerized_feature("race", 16)
    feature_cols = [tf.feature_column.embedding_column(resident_status, 4),
                    tf.feature_column.embedding_column(sex, 2),
                    detail_age,
                    tf.feature_column.embedding_column(marital_status, 5),
                    tf.feature_column.embedding_column(race, 16),]

    estimator = tf.estimator.DNNClassifier(
        feature_columns=feature_cols,
        hidden_units=hidden,
        n_classes=7
    )


    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=trainFile,
        y=trainLabel,
        num_epochs=None,
        shuffle=True
    )

    estimator.train(input_fn=train_input_fn, steps=2000)


    test_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=testFile,
        y=testLabel,
        num_epochs=1,
        shuffle=False)

    accuracy_score = estimator.evaluate(input_fn=test_input_fn)["accuracy"]
    print(accuracy_score, hidden)


    predictInput = pd.DataFrame({"resident_status": [1], "sex":[1], "detail_age": [26], "marital_status": [2], "race": [1]})

    predict_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=predictInput,
        num_epochs=1,
        shuffle=False)

    predictions = estimator.predict(input_fn=predict_input_fn)
    for i, x in enumerate(predictions):
        print(['%.08f' % p for p in x["probabilities"]], x["classes"])
    # print('%.08f' % e for e in x["probabilities"])


    return accuracy_score

def random_hidden():
    return [rnd.randrange(20, 400, 10) for i in range(rnd.randrange(2, 6))]



if __name__ == "__main__":
    trainFile = readFile(TRAIN_FILEPATH)
    testFile = readFile(TEST_FILEPATH)
    trainLabel = trainFile["manner_of_death"]
    trainFile.drop("manner_of_death", axis=1, inplace=True)
    testLabel = testFile["manner_of_death"]
    testFile.drop("manner_of_death", axis=1, inplace=True)
    # trainLabel.ix[:] += 1
    # testLabel.ix[:] += 1
    print(trainLabel.unique(), testLabel.unique())
    while True:
        predict(trainFile, testFile, random_hidden(), trainLabel, testLabel)

