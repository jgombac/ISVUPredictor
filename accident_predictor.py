import pandas as pd
import numpy as np
import tensorflow as tf
from accidents import *


mappings = {
    "VrstaCeste": {},
    "Lokacija": {},
    "VremenskeOkoliscine": {},
    "StanjePrometa": {},
    "StanjeVozisca": {},
    "VrstaVozisca": {}
}

def read_file(filename):
    file = pd.read_csv(filename, sep=";")
    file["SifraOdsekaUlice"] = file["SifraOdsekaUlice"].astype(str)
    #file["UraPN"] = file["UraPN"].astype(str)
    for column in file:
        if file[column].dtype == object:
            file[column] = file[column].map(lambda x: x.strip())
    return file


def map_categories(data):
    for column in mappings:
        print(data[column].unique())
        categories = data[column].astype("category").cat.categories
        codes = data[column].astype("category").cat.codes
        mappings[column] = dict(zip(codes, categories))


def train(features, labels):

    vrsta_ceste = tf.feature_column.categorical_column_with_vocabulary_list(
        "VrstaCeste", features["VrstaCeste"].unique()
    )
    lokacija = tf.feature_column.categorical_column_with_vocabulary_list(
        "Lokacija", features["Lokacija"].unique()
    )
    vreme = tf.feature_column.categorical_column_with_vocabulary_list(
        "VremenskeOkoliscine", features["VremenskeOkoliscine"].unique()
    )
    stanje_prometa = tf.feature_column.categorical_column_with_vocabulary_list(
        "StanjePrometa", features["StanjePrometa"].unique()
    )
    stanje_vozisca = tf.feature_column.categorical_column_with_vocabulary_list(
        "StanjeVozisca", features["StanjeVozisca"].unique()
    )
    vrsta_vozisca = tf.feature_column.categorical_column_with_vocabulary_list(
        "VrstaVozisca", features["VrstaVozisca"].unique()
    )
    sifra_ceste = tf.feature_column.categorical_column_with_vocabulary_list(
        "SifraCesteNaselja", features["SifraCesteNaselja"].unique()
    )
    sifra_ulice = tf.feature_column.categorical_column_with_vocabulary_list(
        "SifraOdsekaUlice", features["SifraOdsekaUlice"].unique()
    )
    v_naselju = tf.feature_column.categorical_column_with_vocabulary_list(
        "VNaselju", features["VNaselju"].unique()
    )
    dan = tf.feature_column.numeric_column("Dan")
    mesec = tf.feature_column.numeric_column("Mesec")

    feature_columns = [dan,
                       mesec,
                       tf.feature_column.indicator_column(v_naselju),
                       tf.feature_column.indicator_column(vrsta_ceste),
                       tf.feature_column.indicator_column(sifra_ceste),
                       tf.feature_column.indicator_column(sifra_ulice),
                       tf.feature_column.indicator_column(lokacija),
                       tf.feature_column.indicator_column(vreme),
                       tf.feature_column.indicator_column(stanje_prometa),
                       tf.feature_column.indicator_column(stanje_vozisca),
                       tf.feature_column.indicator_column(vrsta_vozisca)]
    estimator = tf.estimator.DNNRegressor(
        hidden_units=[100, 200, 100],
        feature_columns=feature_columns,
        model_dir="gombi"
    )

    train_input = tf.estimator.inputs.pandas_input_fn(
        x=features,
        y=labels,
        num_epochs=None,
        shuffle=True
    )

    estimator.train(train_input, steps=500)
    #
    # test_input = tf.estimator.inputs.pandas_input_fn(
    #     x=features,
    #     y=labels,
    #     num_epochs=1,
    #     shuffle=False
    # )
    #
    # accuracy = estimator.evaluate(test_input)
    # print(accuracy)



def predict(features, vocabulary):
    vrsta_ceste = tf.feature_column.categorical_column_with_vocabulary_list(
        "VrstaCeste", vocabulary["VrstaCeste"]
    )
    lokacija = tf.feature_column.categorical_column_with_vocabulary_list(
        "Lokacija", vocabulary["Lokacija"]
    )
    vreme = tf.feature_column.categorical_column_with_vocabulary_list(
        "VremenskeOkoliscine", vocabulary["VremenskeOkoliscine"]
    )
    stanje_prometa = tf.feature_column.categorical_column_with_vocabulary_list(
        "StanjePrometa", vocabulary["StanjePrometa"]
    )
    stanje_vozisca = tf.feature_column.categorical_column_with_vocabulary_list(
        "StanjeVozisca", vocabulary["StanjeVozisca"]
    )
    vrsta_vozisca = tf.feature_column.categorical_column_with_vocabulary_list(
        "VrstaVozisca", vocabulary["VrstaVozisca"]
    )
    sifra_ceste = tf.feature_column.categorical_column_with_vocabulary_list(
        "SifraCesteNaselja", vocabulary["SifraCesteNaselja"]
    )
    sifra_ulice = tf.feature_column.categorical_column_with_vocabulary_list(
        "SifraOdsekaUlice", vocabulary["SifraOdsekaUlice"]
    )
    v_naselju = tf.feature_column.categorical_column_with_vocabulary_list(
        "VNaselju", vocabulary["VNaselju"]
    )
    dan = tf.feature_column.numeric_column("Dan")
    mesec = tf.feature_column.numeric_column("Mesec")

    feature_columns = [dan,
                       mesec,
                       tf.feature_column.indicator_column(v_naselju),
                       tf.feature_column.indicator_column(vrsta_ceste),
                       tf.feature_column.indicator_column(sifra_ceste),
                       tf.feature_column.indicator_column(sifra_ulice),
                       tf.feature_column.indicator_column(lokacija),
                       tf.feature_column.indicator_column(vreme),
                       tf.feature_column.indicator_column(stanje_prometa),
                       tf.feature_column.indicator_column(stanje_vozisca),
                       tf.feature_column.indicator_column(vrsta_vozisca)]
    estimator = tf.estimator.DNNRegressor(
        hidden_units=[100, 200, 100],
        feature_columns=feature_columns,
        model_dir="gombi"
    )
    input = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame(features, index=[0]),
        num_epochs=1,
        shuffle=False
    )
    prediction = estimator.predict(input_fn=input)

    return int(list(prediction)[0]["predictions"][0])

def save_items(features):
        dct = {
            "VrstaCeste": features["VrstaCeste"].unique(),
            "Lokacija": features["Lokacija"].unique(),
            "VremenskeOkoliscine": features["VremenskeOkoliscine"].unique(),
            "StanjePrometa": features["StanjePrometa"].unique(),
            "StanjeVozisca": features["StanjeVozisca"].unique(),
            "VrstaVozisca": features["VrstaVozisca"].unique(),
            "SifraCesteNaselja": features["SifraCesteNaselja"].unique(),
            "SifraOdsekaUlice": features["SifraOdsekaUlice"].unique(),
            "VNaselju": features["VNaselju"].unique(),
            "Dan": features["Dan"].unique(),
            "Mesec": features["Mesec"].unique(),
        }
        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dct.items() ]))
        df.to_csv("input_options.csv")


def load_items(window):
    file = pd.read_csv("input_options.csv")
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
    if len(features) != 11:
        print("You have to select 1 of everything")
        callback("None")
    else:
        features["Dan"] = int(features["Dan"])
        features["Mesec"] = int(features["Mesec"])
        prediction = predict(features, VOCABULARY)
        callback(prediction)


VOCABULARY = None

def init():
    #app = QApplication(sys.argv)
    w = AccidentsDialog(on_predict)
    global VOCABULARY
    VOCABULARY = load_items(w)
    w.show()

    # data = read_file("PN_cleaned.csv")
    # labels = data["UraPN"]
    # features = data.drop(["UraPN"], axis=1)
    # save_items(features)
    # for f, l in zip(features.iterrows(), labels):
    #     print(f[1].to_dict())
    #train(features, labels)

    #sys.exit(app.exec_())

if __name__ == "__main__":
    init()

