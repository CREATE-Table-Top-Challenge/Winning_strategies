import os
import argparse
import shutil
import json
import pandas as pd
from pathlib import Path

IMG_WIDTH = 640
IMG_HEIGHT = 480


def unpackZipFiles(dataLocation, datasetType):
    if datasetType == "train":
        for i in range(1, 9):
            print("Extracting data from: Training_Data_Part{}.zip".format(i))
            extractionDir = os.path.join(dataLocation, "Training_Data_Part{}".format(i))
            if not os.path.exists(extractionDir):
                os.mkdir(extractionDir)
            zipFile = os.path.join(dataLocation, "Training_Data_Part{}.zip".format(i))
            shutil.unpack_archive(zipFile,extractionDir)

    elif datasetType == "test":
        print("Extracting data from: Test_Data.zip")
        extractionDir = os.path.join(dataLocation, "Test_Data")
        if not os.path.exists(extractionDir):
            os.mkdir(extractionDir)
        zipFile = os.path.join(dataLocation, "Test_Data.zip")
        shutil.unpack_archive(zipFile, extractionDir)


def moveFilesToDirectory(videoDir, imageDir, labelDir, labels=None):
    videoID = os.path.basename(videoDir)
    print("Transferring data from video: {}".format(videoID))
    fileNames = os.listdir(videoDir)
    for fn in fileNames:
        oldFileLocation = os.path.join(videoDir, fn)
        ext = os.path.splitext(fn)[-1].lower()
        if ext == ".jpg":
            newFileLocation = os.path.join(imageDir, fn)
            shutil.move(oldFileLocation, newFileLocation)
        else:
            labelFile = pd.read_csv(oldFileLocation)
            try:
                for index, row in labelFile.iterrows():
                    generateLabelFromDict(labelDir, row, labels)
            except KeyError as e:
                print(f"No bounding box labels found. Skipping label conversion for video {fn}.")


def generateLabelFromDict(labelDir, row, labels):
    filename = os.path.join(labelDir, os.path.splitext(row["FileName"])[0] + ".txt")
    with open(filename, "w+") as f:
        bboxes = eval(row["Tool bounding box"])
        if bboxes:
            for bbox in bboxes:
                label_str = bbox["class"]
                label = labels.index(label_str)

                # Convert bbox to YOLOv6 format
                bbox_width = min(int(bbox["xmax"]) - int(bbox["xmin"]), IMG_WIDTH)
                bbox_height = min(int(bbox["ymax"]) - int(bbox["ymin"]), IMG_HEIGHT)
                pixel_center_x = min(int(bbox["xmin"]) + (bbox_width / 2), IMG_WIDTH)
                pixel_center_y = min(int(bbox["ymin"]) + (bbox_height / 2), IMG_HEIGHT)

                x_center = pixel_center_x / IMG_WIDTH
                y_center = pixel_center_y / IMG_HEIGHT
                width = bbox_width / IMG_WIDTH
                height = bbox_height / IMG_HEIGHT

                f.write(f"{label} {x_center} {y_center} {width} {height}\n")


def checkAllPresent(dataLocation, datasetType):
    missingFiles = []
    if datasetType == "train":
        for i in range(1, 9):
            zipFile = os.path.join(dataLocation, "Training_Data_Part{}.zip".format(i))
            if not os.path.exists(zipFile):
                missingFiles.append("Training_Data_Part{}.zip".format(i))
    elif datasetType == "test":
        zipFile = os.path.join(dataLocation, "Test_Data.zip")
        if not os.path.exists(zipFile):
            missingFiles.append("Test_Data.zip")
    if len(missingFiles) > 0:
        print("Missing the following files:")
        for file in missingFiles:
            print("\t{}".format(file))
        exit()


def getLabelsFromCSV(extractDir):
    csv_fns = [path.resolve() for path in Path(extractDir).rglob("*.csv")]
    dataCSVInitialized = False
    for csv in csv_fns:
        labelFile = pd.read_csv(csv)
        if not dataCSVInitialized:
            dataCSV = pd.DataFrame(columns=labelFile.columns)
            dataCSVInitialized = True
        dataCSV = pd.concat([dataCSV, labelFile])
    for column in dataCSV.columns:
        if "Unnamed" in column:
            dataCSV = dataCSV.drop(column, axis=1)
    
    # Encode the labels
    labels = []
    bbox_col = dataCSV["Tool bounding box"].tolist()
    for row in bbox_col:
        for d in eval(row):
            if "class" in d and d["class"] not in labels:
                labels.append(d["class"])
    return labels


def main(FLAGS):
    baseLocation = FLAGS.compressed_location
    skipExtraction = FLAGS.skip_extraction
    datasetType = FLAGS.dataset_type
    classNameJSON = FLAGS.classname_json
    checkAllPresent(baseLocation, datasetType)
    if not skipExtraction:
        unpackZipFiles(baseLocation, datasetType)
    
    # Create data folders, if they don't exist
    image_dir = "data/images"
    label_dir = "data/labels"
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    if datasetType == "train":
        if os.path.exists(classNameJSON) and os.path.getsize(classNameJSON) > 0:
            with open(classNameJSON, "r") as f:
                labels = json.load(f)
                print(labels)
                print(f"Loaded class names from {classNameJSON}")
        else:
            labels = getLabelsFromCSV(baseLocation)
            with open(classNameJSON, "w") as f:
                json.dump(labels, f)
                print(f"Saved class names to {classNameJSON}")
        
        # Train set
        if not os.path.exists(os.path.join(image_dir, "train")):
            os.mkdir(os.path.join(image_dir, "train"))
        if not os.path.exists(os.path.join(label_dir, "train")):
            os.mkdir(os.path.join(label_dir, "train"))
            
        for i in range(1, 8):
            dataFolder = os.path.join(baseLocation, "Training_Data_Part{}".format(i))
            for videoDir in os.listdir(dataFolder):
                moveFilesToDirectory(os.path.join(dataFolder, videoDir), 
                                     "data/images/train", "data/labels/train", labels)
            shutil.rmtree(dataFolder)
            print("Removed empty directory {}".format(dataFolder))
        
        # Validation set
        if not os.path.exists(os.path.join(image_dir, "val")):
            os.mkdir(os.path.join(image_dir, "val"))
        if not os.path.exists(os.path.join(label_dir, "val")):
            os.mkdir(os.path.join(label_dir, "val"))
            
        dataFolder = os.path.join(baseLocation, "Training_Data_Part8")
        for videoDir in os.listdir(dataFolder):
            moveFilesToDirectory(os.path.join(dataFolder, videoDir), 
                                 "data/images/val", "data/labels/val", labels)
        shutil.rmtree(dataFolder)
        print("Removed empty directory {}".format(dataFolder))

    elif datasetType == "test":
        if not os.path.exists(os.path.join(image_dir, "test")):
            os.mkdir(os.path.join(image_dir, "test"))
        if not os.path.exists(os.path.join(label_dir, "test")):
            os.mkdir(os.path.join(label_dir, "test"))
            
        dataFolder = os.path.join(baseLocation, "Test_Data")
        for videoDir in os.listdir(dataFolder):
            moveFilesToDirectory(os.path.join(dataFolder, videoDir), "data/images/test", "data/labels/test")
        shutil.rmtree(dataFolder)
        print("Removed empty directory {}".format(dataFolder))

    else:
        print("Unrecognized dataset type. Must be one of: Train, Test or Unlabelled")
    print("Dataset preparation complete. Data located in directory: {}".format("data"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--compressed_location',
      type=str,
      default='',
      help='Name of the directory where the compressed data files are located'
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Include this argument if data is already extracted"
    )
    parser.add_argument(
      '--dataset_type',
      type=str,
      default='train',
      help='Type of Dataset you are creating: should be train or test'
    )
    parser.add_argument(
      "--classname_json",
      type=str, 
      default="data/class_names.json",
      help="Path of the JSON file for saving the labels"
    )

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
