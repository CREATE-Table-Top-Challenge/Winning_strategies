import os
import cv2
import pandas
import argparse
from YOLOv5 import YOLOv5
from ultralytics import YOLO

def getPredictions(args):
    modelFolder = args.save_location
    # model = YOLOv5()
    # model.loadModel(modelFolder)
    model = YOLO(args.saved_model)
    data_csv = pandas.read_csv(args.data_csv_file)
    predictions = None
    mapping = {0: 'S', 1: 'anesthetic', 2: 'catheter', 3: 'dilator', 4: 'guidewire', 5: 'guidewire_casing', 6: 'nothing', 7: 'scalpel', 8: 'syringe', 9: 'ultrasound'}
    for x in data_csv.index:
        image_file = data_csv["FileName"][x]
        image_path = os.path.join(data_csv["Folder"][x], image_file)
        image = cv2.imread(image_path)
        results = model.predict(image)

        # Format the bounding boxes and save them in the 'Tool bounding box' column
        formatted_results = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bbox = box.xyxy.tolist()
                if (len(bbox)==0):
                    continue  
                bbox = bbox[0]
                cls =  box.cls
                cls_value = mapping[cls.item()]
                conf = box.conf
                conf = conf.item()
                formatted_result = {
                    'class': cls_value,
                    'xmin': bbox[0],
                    'ymin': bbox[1],
                    'xmax': bbox[2],
                    'ymax': bbox[3],
                    'conf':conf
                }
                if conf<0.5: #(cls_value == 'syringe') and conf<0.3
                    continue
                formatted_results.append(formatted_result)
        new_preds = pandas.DataFrame({"FileName":[data_csv["FileName"][x]],
                                      "Tool bounding box":[formatted_results],
                                      "Folder":[data_csv["Folder"][x]]}
                                      )
        if predictions is None:
            predictions = new_preds.copy()
        else:
            predictions = pandas.concat([predictions,new_preds])
        predictions.index = [i for i in range(len(predictions.index))]
    predictions.to_csv(os.path.join(modelFolder,"Results.csv"),index=False)
    
    # predictions_df = pandas.DataFrame(predictions)
    # predictions_df.to_csv(os.path.join(modelFolder, "Results.csv"), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', type=str, default = 'Tool_Detection/yolov8-fintuned1.pt', help="Specify the saved model path")
    
    parser.add_argument(
        '--save_location',
        type=str,
        default='testResult',
        help='Name of the directory where the saved model is located'
    )
    parser.add_argument(
        '--data_csv_file',
        type=str,
        default='dataset\Test_Data\Test_Data.csv',
        help='Path to the csv file containing locations for all data used in testing'
    )

    args, unparsed = parser.parse_known_args()
    getPredictions(args)