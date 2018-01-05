import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QListView, QPushButton, QLabel
from PyQt5.uic import loadUi
from PyQt5 import QtGui as gui

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

class AccidentsDialog(QMainWindow):

    def __init__(self, callback):
        super(AccidentsDialog, self).__init__()
        loadUi("deaths.ui", self)
        predictBtn = self.findChild(QPushButton, "predictBtn")
        fileBtn = self.findChild(QPushButton, "fileBtn")
        predictBtn.clicked.connect(lambda: self.predict_click(callback))


    def fill_list(self, list_name, items):
        list = self.findChild(QListView, list_name)
        model = gui.QStandardItemModel()
        items = self.translateHuman(list_name, items)
        for item in items:
            it = gui.QStandardItem(item)
            model.appendRow(it)
        list.setModel(model)

    def translateHuman(self, column, values):
        return [columns[column][int(value)] for value in values]

    def translateMachine(self, data):
        s = {}
        for key, value in data.items():
            for k, v in columns[key].items():
                if value == v:
                    s[key] = str(k)
        return s

    def predict_click(self, callback):
        self.hide_prediction()
        selected = {}
        for widget in self.findChildren(QListView):
            indexes = widget.selectedIndexes()
            for index in indexes:
                selected[widget.objectName()] = str(index.data())
        callback(self.show_prediction, self.translateMachine(selected))

    def hide_prediction(self):
        label = self.findChild(QLabel, "predictionLbl")
        label.setText("-")

    def show_prediction(self, prediction):
        label = self.findChild(QLabel, "predictionLbl")
        label.setText("ICD koda razloga za smrt: " + str(prediction))

