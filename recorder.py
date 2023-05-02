import numpy as np
import pandas as pd
import csv


headers = ['R hor pupil dist', 'R ver pupil dist',
        'L hor pupil dist', 'L ver pupil dist',
        'R iris cam dist', 'L iris cam dist',
        'x head angle', 'y head angle', 'z head angle',

        'Mouse x', 'Mouse y']

AVERAGE_SIZE = 10

class Recorder:

    def __init__(self):
        self.rolling_average = [None] * AVERAGE_SIZE

        self.dataset = []
        self.averaged_dataset = []
        self.ind = 0

        self.calibration_dataset = []
        self.averaged_calibration_dataset = []


    def addEntry(self, entry):
        self.dataset.append(entry)

        self.rolling_average[self.ind] = entry
        self.ind += 1
        self.ind = self.ind % AVERAGE_SIZE

        if None in self.rolling_average:
            return

        average = np.mean(self.rolling_average, axis=0)
        self.averaged_dataset.append(average)


    def addCalibrationEntry(self, entry):
        self.calibration_dataset.append(entry)




    def export(self):
        with open("dataset.csv","w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(self.dataset)

        with open("averagedDataset.csv","w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(self.averaged_dataset)


    def calibrate(self):
        with open("./model/calibrated.csv","w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(self.calibration_dataset)
