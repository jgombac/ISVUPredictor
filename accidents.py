import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QListView, QPushButton, QLabel
from PyQt5.uic import loadUi
from PyQt5 import QtGui as gui


columns = {
    "StanjeVozisca": {
        "BL": "blatno",
        "MO": "mokro",
        "OS": "ostalo",
        "PN": "poledenelo-neposipano",
        "PP": "poledenelo-posipano",
        "SL": "sneženo-pluženo",
        "SN": "sneženo-nepluženo",
        "SP": "spolzko",
        "SU": "suho",
    },
    "VremenskeOkoliscine": {
        "D": "deževno",
        "J": "jasno",
        "M": "megla",
        "N": "neznano",
        "O": "oblačno",
        "S": "sneg",
        "T": "toča",
        "V": "veter",
    },
    "Lokacija": {
        "C": "cesta",
        "N": "naselje"
    },
    "VrstaCeste": {
        "H": "hitra cesta",
        "L": "lokalna cesta",
        "N": "naselje z ul. sist.",
        "T": "turistična cesta",
        "V": "naselje brez ul. sist.",
        "0": "avtocesta",
        "1": "glavna c. I reda",
        "2": "glavna c. II reda",
        "3": "regionalna c. I reda",
        "4": "regionalna c. II reda",
        "5": "regionalna c. III reda",
    },
    "StanjePrometa": {
        "E": "neznano",
        "G": "gost",
        "N": "normalen",
        "R": "redek",
        "Z": "zastoji",
    },
    "VrstaVozisca": {
        "AH": "hrapav  asfalt/beton",
        "AN": "neraven asfalt/beton",
        "AZ": "zgaljen asfalt/beton",
        "MA": "makadam",
        "OS": "ostalo"
     },
    "VNaselju": {
        "0": "ne",
        "1": "da"
    }
}

classes = {
    "B": "Brez poškodbe",
    "H": "Huda telesna poškodba",
    "L": "Lažja telesna poškodba",
    "S": "Smrt",
    "U": "Brez poškodbe-UZ",
}

class AccidentsDialog(QMainWindow):

    def __init__(self, callback):
        super(AccidentsDialog, self).__init__()
        loadUi("accidents.ui", self)
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
        return [columns[column][value] if column in columns else value for value in values]

    def translateMachine(self, data):
        s = {}
        for key, value in data.items():
            if key not in columns:
                s[key] = value
            else:
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
        label.setText("")

    def show_prediction(self, prediction):
        label = self.findChild(QLabel, "predictionLbl")
        label.setText("Posledice nesreče: " + classes[str(prediction)])

