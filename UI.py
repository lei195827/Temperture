import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from temp2 import MLP
from PyQt5.QtWidgets import QMainWindow, QApplication
import interface
import numpy as np
from joblib import load
from predict2 import load_model, predict_temperature


class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.scaler = None
        self.model = None
        self.device = None
        self.input_boxes = []
        self.ui = interface.Ui_MainWindow()
        self.ui.setupUi(self)
        # "Is_Talking"
        self.input_boxes_names = ["CPU0_T", "CPU1_T", "CPU2_T", "CPU3_T", "Module_humity", "Module_temp", "lm75"]
        self.init()

    def init(self):
        self.ui.clearButton.clicked.connect(self.clear_inputs)
        self.ui.confirmButton.clicked.connect(self.predict_temperature)
        self.input_boxes = self.findChildren(QtWidgets.QLineEdit)

        # 加载保存的scaler对象
        self.scaler = load('scaler.joblib')
        # 加载已经训练好的模型
        self.model, self.device = load_model('model.pth')

    def clear_inputs(self):
        for input_box in self.input_boxes:
            input_box.setText('')

    def predict_temperature(self):
        # get input data from input boxes
        input_data = []
        try:
            for name in self.input_boxes_names:
                input_box = self.findChild(QtWidgets.QLineEdit, name)
                if input_box is not None:
                    input_data.append(float(input_box.text()))
            input_data = np.array(input_data).reshape(1, -1)
            result = input_data[:, :4].sum(axis=1)
            print(f"{input_data}")
            input_data = np.hstack((result.reshape((-1, 1)), input_data[:, 4:]))
            print(f"{input_data}")
            output = predict_temperature(self.model, self.device, input_data, self.scaler)
            output = str(f"{output[0][0]:.3f}") + "摄氏度"
            self.ui.OutText.append(output)
            print(output)  # 22.934   23.32
        except:
            print("输入错误")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
