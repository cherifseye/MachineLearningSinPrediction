from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import os
import numpy as np
import sin_model

class ModelDeployment(QMainWindow):
    def __init__(self):
        super(ModelDeployment, self).__init__()
        self.initUI()
        self.UI()
        self.title1()
        self.prediction()

    def initUI(self):
        self.setWindowTitle('Sinus prediction')
        self.setGeometry(100, 100, 500, 300)

    def UI(self):
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.centralWidget.setObjectName('centralWidget')
    
    def title1(self):
        title1 = QLabel(self.centralWidget)
        title1.setGeometry(QRect(100, 20, 300, 50))
        title1.setText('Sinus prediction')
        title1.setFont(QFont('Arial', 24))
        title1.setStyleSheet('color: #02659D')
        title1.setAlignment(Qt.AlignCenter)

    def prediction(self):
        self.fame = QFrame(self.centralWidget)
        self.fame.setGeometry(QRect(100, 100, 300, 150))
        self.fame.setFrameShape(QFrame.StyledPanel)
        self.fame.setFrameShadow(QFrame.Raised)

        self.input = QLineEdit(self.fame)
        self.input.setGeometry(QRect(20, 20, 250, 30))

        self.button = QPushButton(self.fame)
        self.button.setGeometry(QRect(100, 60, 100, 40))
        self.button.setText('Predict')
        self.button.clicked.connect(self.predict_)

        self.output = QLineEdit(self.fame)
        self.output.setGeometry(QRect(20, 100, 250, 30))
        self.output.setReadOnly(True)

    
    def predict_(self):
        self.input_value = float(self.input.text())
        self.output_value_list = sin_model.predict([[self.input_value]])
        self.output_value = round(self.output_value_list[0], 2)
        self.output.setText(str(self.output_value))





def main():
    app = QApplication(sys.argv)
    ex = ModelDeployment()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

        