import pandas as pd
import tensorflow as tf
import numpy as np
from deaths import *

DEATHS = "umrli.csv"

columns = {
    "leto_dogodka": {
    10: 2010,
    11: 2011,
    12: 2012,
    13: 2013,
    14: 2014,
    15: 2015,
    16: 2016,
},
    "stan": {
        1: "samski/a",
        2: "poročen/a",
        3: "vdovec/a",
        4: "razvezan/a",
        9: "neznano"
    },
    "spol": {
        1: "moški",
        2: "ženski"
    },
    "statisticna_regija": {
        1: "pomurska",
        2: "podravska",
        3: "koroška",
        4: "savinjska",
        5: "zasavska",
        6: "spodnjeposavska",
        7: "jugovzhodna",
        8: "osrednjeslovenska",
        9: "gorenjska",
        10: "notranjo-kraška",
        11: "goriška",
        12: "obalno-kraška",
        99: "neznano"
    },
    "starostna_skupina": {
        1: "0",
        2: "1-4",
        3: "5-9",
        4: "10-14",
        5: "15-19",
        6: "20-24",
        7: "25-29",
        8: "30-34",
        9: "35-39",
        10: "40-44",
        11: "45-49",
        12: "50-54",
        13: "55-59",
        14: "60-64",
        15: "65-69",
        16: "70-74",
        17: "75-79",
        18: "80-84",
        19: "85+"
    },
}

column_mappings = {
    "trans": {
        "lldog": "leto_dogodka",
        "stan": "stan",
        "spol": "spol",
        "statreg": "statisticna_regija",
        "starskup": "starostna_skupina",
        "vzrnassm": "zunanji_vzrok",
        "vzroksmr": "osnovni_vzrok",
    },
    "osnovni_vzrok": {}
}


def read_file(filename):
    file = pd.read_csv(filename, sep=";")
    file.rename(index=str, columns=column_mappings["trans"], inplace=True)
    return file


def transform_data(data):
    return data.replace({x: columns[x] for x in columns})

def training_data(data):
    new_data = data.drop(["zunanji_vzrok"], axis=1)

    raw_labels = new_data["osnovni_vzrok"]

    num_occurrences = pd.DataFrame(raw_labels.value_counts().reset_index())
    num_occurrences.columns = ["value", "occurrences"]
    slice = num_occurrences.head(3)
    print(slice)
    frequent_data = new_data.loc[new_data["osnovni_vzrok"].isin(slice["value"])]

    features = frequent_data.drop(["osnovni_vzrok", "leto_dogodka"], axis=1)
    labels = frequent_data["osnovni_vzrok"].astype(np.str)
    return features, labels



def train(features, labels):
    stan = tf.feature_column.categorical_column_with_vocabulary_list("stan", features["stan"].unique())
    spol = tf.feature_column.categorical_column_with_vocabulary_list("spol", features["spol"].unique())
    statisticna_regija = tf.feature_column.categorical_column_with_vocabulary_list("statisticna_regija", features["statisticna_regija"].unique())
    starostna_skupina = tf.feature_column.categorical_column_with_vocabulary_list("starostna_skupina", features["starostna_skupina"].unique())

    feature_cols = [tf.feature_column.indicator_column(stan),
                    tf.feature_column.indicator_column(spol),
                    tf.feature_column.indicator_column(statisticna_regija),
                    tf.feature_column.indicator_column(starostna_skupina)]

    estimator = tf.estimator.DNNClassifier(
        feature_columns=feature_cols,
        hidden_units=[60, 50, 40],
        n_classes=3,
        label_vocabulary=["I509", "C349", "I640"],
        model_dir="deaths_model"
    )


    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=features,
        y=labels,
        num_epochs=None,
        shuffle=True
    )

    estimator.train(input_fn=train_input_fn, steps=100)


    test_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=features,
        y=labels,
        num_epochs=1,
        shuffle=False)

    accuracy_score = estimator.evaluate(input_fn=test_input_fn)["accuracy"]
    print(accuracy_score)

    return accuracy_score

def predict(features, vocabulary):
    stan = tf.feature_column.categorical_column_with_vocabulary_list("stan", vocabulary["stan"])
    spol = tf.feature_column.categorical_column_with_vocabulary_list("spol", vocabulary["spol"])
    statisticna_regija = tf.feature_column.categorical_column_with_vocabulary_list("statisticna_regija", vocabulary["statisticna_regija"])
    starostna_skupina = tf.feature_column.categorical_column_with_vocabulary_list("starostna_skupina", vocabulary["starostna_skupina"])

    feature_cols = [tf.feature_column.indicator_column(stan),
                    tf.feature_column.indicator_column(spol),
                    tf.feature_column.indicator_column(statisticna_regija),
                    tf.feature_column.indicator_column(starostna_skupina)]

    estimator = tf.estimator.DNNClassifier(
        feature_columns=feature_cols,
        hidden_units=[60, 50, 40],
        n_classes=3,
        label_vocabulary=["I509", "C349", "I640"],
        model_dir="deaths_model"
    )


    input = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame(features, index=[0]),
        num_epochs=1,
        shuffle=False
    )

    prediction = list(estimator.predict(input_fn=input))[0]["classes"][0].decode("utf-8")
    return prediction



def save_items(features):
    dct = {
        "stan": features["stan"].unique(),
        "spol": features["spol"].unique(),
        "statisticna_regija": features["statisticna_regija"].unique(),
        "starostna_skupina": features["starostna_skupina"].unique(),

    }
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dct.items()]))
    df.to_csv("input_options_deaths.csv")


def load_items(window):
    file = pd.read_csv("input_options_deaths.csv")
    vocabulary = {}
    for col in file:
        try:
            items = file[col].unique()
            if pd.isnull(items[-1]):
                items = items[:-1]
            items = sorted(items)
            items = [str(int(x)) if isinstance(x, float) else str(x) for x in items]
            vocabulary[col] = items
            window.fill_list(col, items)
        except:
            pass
    return vocabulary


def on_predict(callback, features):
    if len(features) != 4:
        print("You have to select 1 of everything")
        callback("None")
    else:
        prediction = predict(features, VOCABULARY)
        callback(prediction)


VOCABULARY = None

def init():
    # data = read_file(DEATHS)
    # features, labels = training_data(data)

    #
    # train(features, labels)
    # save_items(features)
    #app = QApplication(sys.argv)
    w = AccidentsDialog(on_predict)
    global VOCABULARY
    VOCABULARY = load_items(w)
    w.show()

    #sys.exit(app.exec_())


if __name__ == "__main__":
    init()