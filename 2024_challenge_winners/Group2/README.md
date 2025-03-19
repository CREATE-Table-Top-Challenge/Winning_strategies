# Group#2
  
## Team members:
Jianming Yang, Kaitlyn Kobayashi, Flourish Adebayo, Dumitru Cernelev (Mentor: Nooshin Maghsoodi)
  
## Strategy:
#### General approach

**How did you approach the challenge overall? Did you target one particular subtask over the others?**
At the outset of the challenge, we collectively decided to address all three subtasks with equal priority.

**How did you divide the work to be done amongst all team members?**
Initially, we assigned two team members to tackle subtask 1, two to handle subtask 2, one for subtask 3, and the final member responsible for data labeling. However, due to unforeseen circumstances, one member exited the challenge, prompting the subtask 3 participant to focus to both subtask 3 and subtask 1. Initially, each pair of members in subtasks 1 and 2 divided responsibilities, with one handling baseline implementation and experiments, while the other conducted substitution method research. Mean while, one of the other members also helped for annotation.

**How did you make use of the resources provided (e.g. base-line code)**
For subtasks 1 and 2, we used the provided baselines, utilizing them as reliable fallback models.

#### Sub-task 1: Tool detection
**What was your general strategy for approaching this subtask?**
We opted to transition from the baseline to implementing YOLOv8 after assessing initial results.

**How did you come up with your solution for the subtask?**
Leveraging Utralytics, we seamlessly integrated a pretrained YOLOv8 model into YOLOv5.py. Subsequently, we focused on incorporating augmentation parameters, resizing images to 640, adjusting epochs and batch size, and refining the data split for training and validation.

**Did you modify/optimize the training process in any way?**
Yes, we enhanced the training process by utilizing a pretrained YOLOv8 model instead of YOLOv5.

**How did you decide to split the data for training?**
We allocated 20% of the videos from the beginning indices for validation, with the remainder reserved for training.

**Did you use any preprocessing or postprocessing? If so, what?**
We implemented data augmentation and balancing techniques as preprocessing methods and also as postprocessing the confidence was used for filtering the boundry boxes with confidence less than 0.5.

H**ow much time did you dedicate to this task (approximately)?**
In total, we dedicated approximately one week to this task.


#### Sub-task 2: Workflow recognition
What was your general strategy for approaching this subtask? 
We had two team members (Jianming and Kaitlyn) tackle this subtask together to leverage their knowledge in computer vision and data analysis. We investigated the dataset prior to training and tested various hyperparameter combinations based on that investigation. We used unlabelled videos which had completed annotations to get predictions. We compared the ground truth annotations to the predictions to see if there were error patterns we could fix using post-processing. After comparing the performance of all the models, we used the hyperparameters that yielded the best validation performance to retrain the model using the training data combined with all the newly labelled data provided to us over the challenge to get our final predictions. 

How did you come up with your solution for the subtask? 
The first thing we looked at was the dataset to see what the balance of labels were: 
![image](https://github.com/CREATEGroup2/Challenge_Code/assets/79864646/b835c235-ff69-43e1-9947-bfa285a31ef6)

Once we saw there was a class imbalance, we used augment_cnn=True and balance_lstm=True to try and handle this problem. This had improved performance over the baseline model. Once we knew this, we began changing the validation percentages to 20%, 15%, and 10%. After comparing validation performances of all models, we found that the combination of augment_cnn=True, balance_lstm=True, and validation_percentage=0.1 worked best. All other parameters were set to their default values.

Once more data was available to us through annotating the unlabelled data, we retrained the model using the training data and newly annotated data to obtain our final model. 

Did you modify/optimize the training process in any way? 
The only optimizations we made were changing the hyperparameters as described above and adding more data during training. 

How did you decide to split the data for training?  
Choosing the training and validation split was done through trial and error and comparing model validation performance. We decided to use annotated videos from the Unlabelled dataset as our hold-out testing set to see what the predictions would be prior to retraining our model with them. 

Did you use any preprocessing or postprocessing? If so, what?  
We tried to add a augmentation variation by tilting images to simulate different off-plane angles a tool may enter the frame, this introduced 1.5% more accuracy than the baseline models. We did smoothing and pattern-correction for post-processing.

How much time did you dedicate to this task (approximately)? 
The two members dedicated about 1.5 weeks to this task. For one of the two members who wanted to try different models - failed and wasted 1 other week. :<

#### Sub-task 3: Ultrasound segmentation
**What was your general strategy for approaching this subtask?**
Our strategy involved utilizing SAM as the foundational model and integrating it with another open-source model to generate bounding boxes for input data.

**Which foundation model did you use and why?**
We chose SAM due to its pretraining on a large dataset and its efficacy in zero-shot inference. Additionally, we employed YOLOv8 to generate bounding boxes as input for SAM.

**Did you fine-tune the model?**
If so, how did you do this? Yes, we fine-tuned YOLOv8 using bounding boxes solely from the Ultrasound class in the training data.

**Did you make use of the existing labels?**
If so, how did you incorporate them? We utilized the existing labels to fine-tune YOLOv8 specifically for identifying ultrasound bounding boxes.

**Did you use any preprocessing or postprocessing?** **If so, what? **
No preprocessing or postprocessing techniques were utilized in this approach.

**How much time did you dedicate to this task (approximately)?**
Approximately 2 weeks were dedicated to this task.

#### Unlabelled Data
How did you approach using the unlabelled data?  
We followed the instructions on the challenge GitHub page. The setup involved installing the necessary libraries, starting with PyTorch. Additional required libraries included ultralytics, torchvision, transformers, PyQt6, superqt, and sklearn. Some libraries like numpy and pandas were already satisfied. After installing everything, we launched the tool using python <path to repo>/Automated_Annotator.py. Key commands include navigating images (n/p), cycling display modes (m), and exporting bounding boxes (Ctrl+e).

Did you use it during the training process?  The unlabelled data was utilized in the training process.

What tools did you choose to use to annotate/review the labels?  To annotate and review the labels, we utilized the following tools:

          - Qt Autoannotater (https://github.com/JianmingY/Qt_Autoannotater)
          - Automated Annotator (https://github.com/RebeccaHisey/Automated_Annotator)

How much time did you dedicate to this task (approximately)?  This task lasted about two weeks. 

## Setup:
How did you set up your environment? Include detailed steps. If you used the environment that was provided, you may simply refer back to the original challenge page.  
We followed the instructions on the challenge GitHub page 

If you used the provided environment, did you install any additional libraries? Include steps for installation (can be a link to existing documentation) 

For subtasks 1 and 2 the same instruction of the baseline will work without installing additional packages and for subtask 3 just SAM is needed. 
You can install sam with the following command:
```
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
```
Also, you need to download the SAM checkpoint and put it in Probe_Segmentation foler:
```
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
## Training the networks:

#### Sub-task 1: Tool detection
Just run Train_YOLOv8.py with the same command as baseline.
#### Sub-task 2: Task Recognition
Run the following line with the training data and newly annotated data from the Unlabelled dataset in one csv: 
```
python C:/Users/SampleUser/Documents/Central_Line_Challenge/Task_Recognition/Train_CNN_LSTM.py --save_location=C:/Users/SampleUser/Documents/taskDetectionRun1 --data_csv_file=C:/Users/SampleUser/Documents/Training_Data/Training_Data.csv --augment_cnn=True --balance_lstm=True --validation_percentage=0.1
```
#### Sub-task 3: Ultrasound probe segmentation
To produce the fine-tuned YOLOv8 model for boundary detection, we generated the TXT label files containing only the ultrasound boxes. After producing the required dataset, the Tool Detection model trained using this dataset as subtask 1. The best model obtained from this training step serves as our model for producing bounding boxes for the segmentation task. Please note that a pre-trained model is provided with the code, eliminating the need to re-run this step. After procucing the model, since we use SAM in zero-shot-inference mode, we don't need any other training. To evaluate this model you can use the following command.

To run the foundation model, simply execute the YOLOv_SAM.py as following:
```
python ProbSegmentation/YOLOv_SAM.py --path=/path/to/input --output_dir=/path/to/output
```

## Generating test predictions
Note that your code must take in the dataset using the same csv format as the baseline networks provided and produce results that are correctly formatted for submission. No restructuring of the data folders is permitted.
#### Sub-task 1: Tool detection
Test predictions were generated using the GPU server and the methods from the CREATE Central Line Challenge baseline instructions:
```
python Tool_Detection/generateTestPredictions.py --save_location=/path/to/save_loc --data_csv_file=/path/to/Test_Data/Test_Data.csv
```
  
#### Sub-task 2: Task Recognition
Include steps on how to produce predictions on test results  
After CNN-LSTM model prediction. Run following to post-process prediction.
```
python Post_processing.py -csv main_csv_file_path
```
#### Sub-task 3: Task Recognition

```
python ProbSegmentation/YOLOv_SAM.py --path=/path/to/input --output_dir=/path/to/output
```

## Final reflections
Include a short reflection on your general strategy.  
We had discussed our strengths, weaknesses, and experience at the beginning of the challenge and tasks were delegated as appropriate. To be more efficient, we had a member run Subtask 2 (since it took the least time to train) on their own computer to free the GPU server for other subtasks. 

Do you think that your strategy worked well?  
For the most part, yes. We were all able to contribute to the challenge regardless of our backgrounds and we were able to generate results. 

What would you do differently next time?
Split the task of doing annotations evenly amongst the team. Having more data to train on really makes a big difference in model performance, but it can be very time consuming and frustrating for one or two team members to do. It also inherantly gives more insight into the dataset. 
Have a better understanding of team member's time limitations so we can delegate an appropriate amount of work. It is a stressful time of year with thesis work and final projects for classes, so knowing how much time each team member could spare for working on this project would have made managing communication and work expectations easier.

Do you have any advice for future participants of the challenge?  
Get the baselines running right away so you have a backup plan if your new models don't end up working. 
Do the annotations. It's worth it for understanding the data and also for the points. 
Communicate boundaries and your availability to your whole team. 
If you can, run some models locally to free up the GPU resource given for this challenge. 
