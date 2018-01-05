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
    file = pd.read_excel(filename, sheetname="List2")
    file["SifraOdsekaUlice"] = file["SifraOdsekaUlice"].astype(str)
    file["SifraCesteNaselja"] = file["SifraCesteNaselja"].astype(str)
    file["VrstaCeste"] = file["VrstaCeste"].astype(str)
    file["UraPN"] = file["UraPN"].astype(np.int32)
    file["Dan"] = file["UraPN"].astype(np.int32)
    file["Mesec"] = file["UraPN"].astype(np.int32)
    for column in file:
        if file[column].dtype == object:
            try:
                file[column] = file[column].map(lambda x: x.strip())
            except Exception as e:
                print(column, e)
                exit(0)
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
    ura = tf.feature_column.numeric_column("UraPN")
    dan = tf.feature_column.numeric_column("Dan")
    mesec = tf.feature_column.numeric_column("Mesec")

    feature_columns = [ura,
                       dan,
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
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[100, 200, 100],
        feature_columns=feature_columns,
        n_classes=5,
        label_vocabulary=["B", "H", "L", "S", "U"],
        model_dir="accidents_model"
    )

    train_input = tf.estimator.inputs.pandas_input_fn(
        x=features,
        y=labels,
        num_epochs=None,
        shuffle=True
    )

    estimator.train(train_input, steps=500)

    test_input = tf.estimator.inputs.pandas_input_fn(
        x=features,
        y=labels,
        num_epochs=1,
        shuffle=False
    )

    accuracy = estimator.evaluate(test_input)
    print(accuracy)



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
    ura = tf.feature_column.numeric_column("UraPN")
    dan = tf.feature_column.numeric_column("Dan")
    mesec = tf.feature_column.numeric_column("Mesec")

    feature_columns = [ura,
                       dan,
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
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[100, 200, 100],
        feature_columns=feature_columns,
        n_classes=5,
        label_vocabulary=["B", "H", "L", "S", "U"],
        model_dir="accidents_model"
    )
    input = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame(features, index=[0]),
        num_epochs=1,
        shuffle=False
    )
    #prediction = list(estimator.predict(input_fn=input))
    prediction = list(estimator.predict(input_fn=input))[0]["classes"][0].decode("utf-8")
    print(prediction)
    #return int(list(prediction)[0]["predictions"][0])
    return prediction

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
            "UraPN": features["UraPN"].unique(),
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
    if len(features) != 12:
        print("You have to select 1 of everything")
        callback("None")
    else:
        features["UraPN"] = int(features["UraPN"])
        features["Dan"] = int(features["Dan"])
        features["Mesec"] = int(features["Mesec"])
        prediction = predict(features, VOCABULARY)
        callback(prediction)


VOCABULARY = None

def num_occur(data):

    raw_labels = data["KlasifikacijaNesrece"]

    num_occurrences = pd.DataFrame(raw_labels.value_counts().reset_index())
    num_occurrences.columns = ["value", "occurrences"]
    #slice = num_occurrences.head(3)
    print(num_occurrences)

def init():
    #data = read_file("PN.xlsx")
    #num_occur(data)
    #exit()

    # app = QApplication(sys.argv)
    w = AccidentsDialog(on_predict)
    global VOCABULARY
    VOCABULARY = load_items(w)
    w.show()


    #print(data)
    #labels = data["KlasifikacijaNesrece"]
    #features = data.drop(["KlasifikacijaNesrece"], axis=1)
    #save_items(features)
    # for f, l in zip(features.iterrows(), labels):
    #     print(f[1].to_dict())
    #train(features, labels)

    # sys.exit(app.exec_())

if __name__ == "__main__":
    init()

