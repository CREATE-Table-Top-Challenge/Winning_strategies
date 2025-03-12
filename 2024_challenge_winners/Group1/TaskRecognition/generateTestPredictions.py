import os
from PIL import Image
import sys
import numpy
import pandas
pandas.options.mode.chained_assignment = None
import CNN
import argparse

FLAGS = None

class Predict_CNN:
    def getPredictions(self):
        network = CNN.CNN()
        if FLAGS.save_location == "":
            print("No save location specified. Please set flag --save_location")
        elif FLAGS.data_csv_file == "":
            print("No dataset specified. Please set flag --data_csv_file")
        else:
            self.saveLocation = FLAGS.save_location
            self.networkType = "CNN"
            self.dataCSVFile = pandas.read_csv(FLAGS.data_csv_file)
            network.loadModel(self.saveLocation)
            network.cnn_model.cuda("cuda")
            numClasses = network.num_classes
            num_features = network.num_features
            print(numClasses)
            for task in network.task_class_mapping:
                if network.task_class_mapping[task]=="nothing":
                    nothingIndex = task
            columns =["FileName", "Time Recorded","Overall Task"]# ["FileName", "Time Recorded","Overall Task"] #+ [network.task_class_mapping[i] for i in range(network.num_classes)]
            predictions = pandas.DataFrame(columns=columns)
            predictions["FileName"] = self.dataCSVFile["FileName"]
            predictions["Time Recorded"] = self.dataCSVFile["Time Recorded"]
            for i in self.dataCSVFile.index:
                if i%500 == 0 or i==len(self.dataCSVFile.index)-1:
                    print("{}/{} predictions generated".format(i,len(self.dataCSVFile.index)))
                image = Image.open(os.path.join(self.dataCSVFile["Folder"][i],self.dataCSVFile["FileName"][i]))
                taskPrediction = network.predict(image)
                taskLabel,confidences = taskPrediction.split('[[')
                predictions["Overall Task"][i] = taskLabel
            predictions.to_csv(os.path.join(self.saveLocation,"Task_Predictions.csv"),index=False)
            print("Predictions saved to: {}".format(os.path.join(self.saveLocation,"Task_Predictions.csv")))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--save_location',
      type=str,
      default='',
      help='Name of the directory where the saved model is located'
  )
  parser.add_argument(
      '--data_csv_file',
      type=str,
      default='',
      help='Path to the csv file containing locations for all data used in testing'
  )


FLAGS, unparsed = parser.parse_known_args()
tm = Predict_CNN()
tm.getPredictions()