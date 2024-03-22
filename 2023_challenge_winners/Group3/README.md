# Challenge_Code
This repository should include all materials needed to reproduce your results file. Please fill in the following sections to describe your approach to the challenge and how to set up your environment and run your code. Minimum questions to be answered are listed in each section, but feel free to add more details beyond what's listed. The information included in this file, is also what is expected in your final results presentation.  
  
This ReadME file must be complete by the results presentation on May 5th, 2023 to be eligible for awards.    
  
Winning teams (for each subtask and overall) will have their submission featured in the Winning_Strategies repository on the main challenge page for all future participants to see.  
  
## Team members:
Dilakshan Srikanthan, Alex Robins, Chris Yeung, and Joeana Cambranis-Romero

## Challenge Background:
Receiving feedback is essential for trainees learning how to perform surgical procedures, such as central venous catheterization (CVC). However, physicians often have competing priorities that make it difficult to provide detailed feedback to trainees. Therefore, the opportunity to take advantage of an automatic feedback system helps alleviate pressure from physicians. An example of an automatic system used for training is known as the Perk Tutor, which uses real-time needle positioning and ultrasound imaging (1). Residents that used the Perk Tutor demonstrated better performance than those using only ultrasound for a lumbar puncture simulated procedure (1). Central Line Tutor is another example of an automatic system that can be used for practicing CVC procedures (2). Research has shown that Central Line Tutor is able to successfully determine relevant transition points in real-time webcam video for workflow detection purposes (2). Another task that can be aided through computer-assisted training involves usage of surgical tools, which means that the system should be able to perform object detection to identify the different tools. A Faster Region-Based Convolutional Neural Network was applied to video recordings of CVC procedures and the model was able to successfully identify the surgical tools (3). The goal of the work presented here is to improve the existing Central Line Tutor platform’s ability to recognize the workflow and detect the surgical tools and their relative positioning.

## Strategy:
#### General approach
How did you approach the challenge overall? Did you target one particular subtask over the others?    
  The team had a brief meeting on the first day to brainstorm and divide the tasks trying to give the same importance to all of the subtasks. 

How did you divide the work to be done amongst all team members?    
  The two memebrs with more experience started working on subtasks 1 and 2. 
  One memeber was in charge of reviweing the data provided and label the unlabelled data. 
  Finally the last memebr was in charged of the reasearch part of the challenge. 
  
How did you make use of the resources provided (e.g. base-line code)  
  The proivded GPU was used for training. 
  

#### Sub-task 1: Tool detection
How did you come up with your solution for the subtask? 
  Our team believed that applying another YOLO model for the tool detection subtask was the best approach. Single stage object detectors, such as YOLO, have comparable detection accuracies to two stage object detectors; however the inference time is much shorter (4). The baseline model takes advantage of YOLO v3, which uses a Darkent-53 framework consisting of 53 convolutional layers (4). We implemented a YOLO v6 model, which has a different architecture where the head is decoupled and additional layers are present (5). These features allow the model to generate higher mAP scores (5).

Did you modify/optimize the training process in any way?
  For the YOLOv6 architecture, the backbone uses CSPBepBackbone, the neck uses CSPRepBiFPANNeck, and the head uses EffiDeHead. The SGD optimizer was used. And the GIOU type of IOU was implemented. The batch size was 32, whereas the baseline model only used a batch size of 4. 

How did you decide to split the data for training?  
  Parts 1-7 were used for training and Part 8 was used for validation.
  
Did you use any preprocessing or postprocessing? If so, what?  
  Yes, random flips and rotations. 
  
How much time did you dedicate to this task (approximately)?  
  Approximately 3 days were dedicated to this task. 

#### Sub-task 2: Workflow recognition
How did you come up with your solution for the subtask?

We decided to use the baseline network and perform hyperparameter tuning to optimize model performance. We tried learning rates of 0.001, 0.0001, and 0.00001 and different number of neurons in the Dense layer of the CNN. To perform the search, we used the HyperBand algorithm from the Keras Tuner library. The algorithm works by training a large number of models for a small number of epochs in a bracket-like format. Only the best performing models move on to the next round, essentially halving the number of models to train each trial.

Did you modify/optimize the training process in any way? No

How did you decide to split the data for training? Same as in the baseline

Did you use any preprocessing or postprocessing? If so, what? Same as in the baseline

How much time did you dedicate to this task (approximately)? Approximately 2 days

#### Unlabelled Data
How did you approach using the unlabelled data?  
  First all data was downloaded. The approach was as follows:  With the labels open the data was reviewed, with the purpose of understanding how each procedure tasks was performed and labelled (bounding boxes) in the labelled data provided. Once understood, 3D Slicer was used to select the region of interest (bounding box) sizes of each tool and the label format (excel file) was filled out (with tool visible, tasks and bounding box information). 

Did you use it during the training process?  
  No, not enough time.
  
What tools did you choose to use to annotate/review the labels?  
  The open source 3D Slicer was used to review the data and obtained the bounding box information. 
  
How much time did you dedicate to this task (approximately)?  
  Being a large data size (aprox 500 2D slices) ~2 days were employed to label just one folder of the unlabelled data, plus half a day for reviweing the labelled data to understand the bounding box process. 

## Setup:
How did you set up your environment? Include detailed steps. If you used the environment that was provided, you may simply refer back to the original challenge page.  
If you used the provided environment, did you install any additional libraries? Include steps for installation (can be a link to existing documentation)

## Training the networks:
Note that your networks must take in the dataset using the same csv format as the baseline networks provided. No restructuring of the data folders is permitted.
#### Sub-task 1: Tool detection

First, clone the repo and navigate to the YOLO directory:

```
git clone https://github.com/CREATEGroup3/Challenge_Code.git
cd detection/YOLOv6
```

Install dependencies:

```
pip install -r requirements.txt
```

Prepare dataset in YOLOv6 format:

```
python prepareYOLODataset.py --compressed_location /home/jovyan/Downloads --dataset_type train
```

Arguments:
- `--compressed_location`, directory where compressed data files are stored
- `--dataset_type`, which dataset to preprocess, train or test
- `--classname_json`, json file for saving list of class names corresponding with integer label (will load if file exists), default="data/class_names.json"
- `--skip_extraction`, include this argument if data has already been extracted (but not preprocessed)

Train the network:

```
python tools/train.py --batch 32 --conf configs/yolov6s_finetune.py --data data/dataset.yaml --fuse_ab --device 0 --epochs 10
```

The full list of available arguments can be found [here](https://github.com/CREATEGroup3/Challenge_Code/blob/e587f4686baea2cadcff096d592a60f860021174/detection/YOLOv6/tools/train.py "tools/train.py").

Please also ensure that the order and number of class names in `data/dataset.yaml` matches those in `data/class_names.json`.

#### Sub-task 2: Task Recognition

Training this network is almost identical to the baseline code. Skip the following command if the training data has already been loaded using the baseline code. Replace paths as necessary. From the root directory of the repository:

```
python recognition/prepareDataset.py --compressed_location=/home/jovyan/Downloads --target_location=/home/jovyan/Documents/Training_Data --dataset_type=Train
```

Train the network:

```
python recognition/Train_CNN_LSTM.py --save_location /home/jovyan/Documents/TaskRecognitionRun1 --data_csv_file /home/jovyan/Documents/Training_Data/Training_Data.csv
```

The same optional arguments from the [baseline](https://github.com/CREATE-Table-Top-Challenge/Central_Line_Challenge/tree/a66085fbcdc2c90f7e0f50bc40df0c76e747c8ac#optional-flags "Optional flags") can be used here.

## Generating test predictions
Note that your code must take in the dataset using the same csv format as the baseline networks provided and produce results that are correctly formatted for submission. No restructuring of the data folders is permitted.
#### Sub-task 1: Tool detection

Make sure you are in the YOLOv6 directory:

```
cd detection/YOLOv6
```

Extract the test data:

```
python prepareYOLODataset.py --compressed_location /home/jovyan/Downloads --dataset_type test
```

Use the best trained YOLOv6 model to make infer bounding boxes on the test images:

```
python tools/infer.py --yaml data/dataset.yaml --weights runs/train/exp/weights/best_ckpt.pt --source data/images/test --device 0 --save-txt
```

The full list of available arguments can be found [here](https://github.com/CREATEGroup3/Challenge_Code/blob/7246d97a48ad88df1f0451f37aea9cd352752582/detection/YOLOv6/tools/infer.py "tools/infer.py").

Once inference is complete, the images with overlayed bounding boxes should be in `detection/YOLOv6/runs/inference/exp/test` and the text files containing the bounding boxes and confidence can be found in `detection/YOLOv6/runs/inference/exp/test/labels`.

To generate the bounding boxes in the submission format, run the [`generateSubmissionBboxes.py`](https://github.com/CREATEGroup3/Challenge_Code/blob/7f9543a7088463daa277edb04f7643c07ccf38e7/detection/YOLOv6/generateSubmissionBboxes.py "detection/YOLOv6/generateSubmissionBboxes.py") script:

```
python generateSubmissionBboxes.py --label-dir runs/inference/exp/test/labels
```

Arguments:
- `--label-dir`, directory where label text files from previous step are saved
- `--label-csv`, path of csv file to save output bounding boxes to, default="results/test_labels.csv"
- `--classname-json`, json file of class names corresponding with integer label, default="data/class_names.json"
- `--img-size`, image dimensions in pixels, default=[480, 640]

The resulting csv file only contains the image filename and the bounding boxes. For proper submission, it must be combined with the csv file generated from subtask 2.

#### Sub-task 2: Task Recognition

Prepare dataset:

```
python recognition/prepareDataset.py --compressed_location /home/jovyan/Downloads --target_location /home/jovyan/Documents --dataset_type Test
```

From repository root directory:

```
python recognition/generateTestPredictions.py --save_location /home/jovyan/Documents/TaskRecognitionRun1 --data_csv_file /home/jovyan/Documents/Test_Data/Test_Data.csv
```

#### Generating final submission:

After both csv files have been generated, combine them using the following command to produce the submission csv file:

```
python mergePredictionCsvs.py --bbox-csv detection/YOLOv6/results/test_labels.csv --task-csv TaskRecognitionRun1/Task_Predictions.csv --results-csv Group_3_results.csv
```

## Final reflections
Include a short reflection on your general strategy.

Do you think that your strategy worked well?

#### Subtask 1 Results:

- mAP (IoU 50%): 0.47834503311616294 
- mAP (IoU 75%): 0.34995143539881024
- mAP (IoU 90%): 0.08118955669761224

Subtask 1 overall results: 0.91

#### Subtask 2 Results:

- Class-weighted accuracy: 0.69 
- Average precision: 0.68
- Average recall: 0.50
- Starting transitional delay: 0.47
- End transitional delay: 0.52

Subtask 2 overall results: 0.88

Annotation score: 0.0

Overall score: 1.78

What would you do differently next time?  
Do you have any advice for future participants of the challenge? 

## References
1. Keri, Zsuzsanna & Sydor, Devin & Ungi, Tamas & Holden, Matthew & McGraw, Robert & Mousavi, Parvin & Borschneck, Dan & Fichtinger, Gabor & Jaeger, Melanie. (2015). Computerized training system for ultrasound-guided lumbar puncture on abnormal spine models: a randomized controlled trial. Canadian Anaesthetists? Society Journal. 62. 10.1007/s12630-015-0367-2.

2. Rebecca Hisey, Tamas Ungi, Matthew Holden, Zachary Baum, Zsuzsanna Keri, Caitlin McCallum, Daniel W. Howes, Gabor Fichtinger, "Real-time workflow detection using webcam video for providing real-time feedback in central venous catheterization training," Proc. SPIE 10576, Medical Imaging 2018: Image-Guided Procedures, Robotic Interventions, and Modeling, 1057620 (13 March 2018); https://doi.org/10.1117/12.2293494

3. Olivia O'Driscoll, Rebecca Hisey, Daenis Camire, Jason Erb, Daniel Howes, Gabor Fichtinger, Tamas Ungi, "Object detection to compute performance metrics for skill assessment in central venous catheterization," Proc. SPIE 11598, Medical Imaging 2021: Image-Guided Procedures, Robotic Interventions, and Modeling, 1159816 (15 February 2021); https://doi.org/10.1117/12.2581889

4. Diwan, T., Anirudh, G. & Tembhurne, J.V. Object detection using YOLO: challenges, architectural successors, datasets and applications. Multimed Tools Appl 82, 9243–9275 (2023). https://doi.org/10.1007/s11042-022-13644-y

5. Nelson, J. (2021, June 7). Your Comprehensive Guide to the YOLO Family of Models. Roboflow. Retrieved May 1, 2023, from https://blog.roboflow.com/guide-to-yolo-models/ 

