import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QListView, QPushButton, QLabel
from PyQt5.uic import loadUi
from PyQt5 import QtGui as gui


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
        for item in items:
            it = gui.QStandardItem(item)
            model.appendRow(it)
        list.setModel(model)

    def predict_click(self, callback):
        self.hide_prediction()
        selected = {}
        for widget in self.findChildren(QListView):
            indexes = widget.selectedIndexes()
            for index in indexes:
                selected[widget.objectName()] = str(index.data())
        print(selected)
        callback(self.show_prediction, selected)

    def hide_prediction(self):
        label = self.findChild(QLabel, "predictionLbl")
        label.setText("-")

    def show_prediction(self, prediction):
        label = self.findChild(QLabel, "predictionLbl")
        label.setText("Predicted cause of death: " + str(prediction))

