import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QListView, QPushButton, QLabel
from PyQt5.uic import loadUi
from PyQt5 import QtGui as gui

import tensorflow
import numpy
import pandas

import death_predictor as dt
import accident_predictor as ac

class MainDialog(QMainWindow):

    def __init__(self):
        super(MainDialog, self).__init__()
        loadUi("entry.ui", self)
        accidentsBtn = self.findChild(QPushButton, "accidentsBtn")
        deathsBtn = self.findChild(QPushButton, "deathsBtn")
        accidentsBtn.clicked.connect(self.show_accidents)
        deathsBtn.clicked.connect(self.show_deaths)

    def show_accidents(self):
        dt.init()

    def show_deaths(self):
        ac.init()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    entry = MainDialog()
    entry.show()
    sys.exit(app.exec_())