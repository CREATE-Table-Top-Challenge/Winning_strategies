# Challenge_Code
This repository should include all materials needed to reproduce your results file. Please fill in the following sections to describe your approach to the challenge and how to set up your environment and run your code. Minimum questions to be answered are listed in each section, but feel free to add more details beyond what's listed. The information included in this file, is also what is expected in your final results presentation.  
  
This ReadME file must be complete by the results presentation on May 5th, 2023 to be eligible for awards.    
  
Winning teams (for each subtask and overall) will have their submission featured in the Winning_Strategies repository on the main challenge page for all future participants to see.  
  
## Team members:
Mohamed Harmanani, Kaitlyn Kobayashi, Aden Wong, Rachel Theriault, Edward Wang
  
## Strategy:
#### General approach
How did you approach the challenge overall? Did you target one particular subtask over the others?
  - Since we were provided with base models that have successfully been applied to each subtask, we decided to approach this challenge by optimizing hyperparameters, and training/validation splits.
   - We started training models as early as possible so we could develop labels for the unlabelled data that were then edited by the group members to maximize our score distribution acorss the categories.
  
How did you divide the work to be done amongst all team members?
  - We first determined who on the team had experience using the python environment for deep learning development. Next, we determined who would be in charge of using the GPU server, and who had access to extra GPUs so we could train the models for each task simultaneously. 
  - All team members were involved in strategy development and determining which training runs of the models were priority
  - Mohamed had experience in Python DL development so he implemented the first code change: updating the training validation split such that 1 expert and 1 novice video were used as validation for each model. After updating the code, Mohamad was also in charge of training the CNN and LSTM for subtask 2 with updates made to the learning rate determined by the group.
  - Kaitlyn was in charge of using the provided virtual machine to train the models. This involved determining how to set-up specificiations for the virtual machine, create a server for our team, and how to run the correct files for each subtask. While the models were running, Kaitlyn also took the lead on developing code to evaluate the results for subtask 2 so we could get an idea of what our score woould be for the subtask.
  - Rachel was in charge of balancing the training data for each epoch of subtask 1 model training. Rachels focus was on balancing the amount of frames containing each tool in each epoch. We did not have time to train the model with this update.
  - Adan was also in charge of balancing the training data  with a focus on balancing the amount frames per participant in each epoch. 
  - Edward was in charge of updating the anchor boxes for subtask 1. He used and modified the provided code for this task.
  - All group members also particpated in reviewing task recognition annotations for the unlabelled data.

How did you make use of the resources provided (e.g. base-line code)  
  - We began with training the provided model using the provided base-line code and all modifications made were extensions of the provided code.
  - Most modifications to the baseline code were generator files for each subtask to adjust training/validation split as well as creating balanced datasets in each epoch.
  - Some modifications the config files were made when adjusting hyperparameters
  - We used Rebecca's annotation reviewer to explore the dataset, and learn how to modify the predicted labels for the unlabelled datasets.
  - We used the provided training data to train our models, and to learn what correct labels for task recognition looked like
  - We used the provided GPU server to train the tool recognition model, task recognition was trained on a separate device.

**Figures**
We show below the training curves for one of the 8 models trained as part of the ensemble used for this challenge. This model was trained on folds AN01-04 and MS01-03, with AN04 used as validation.  

> Train and validation Accuracy per epoch for the CNN model

<img width="516" alt="image" src="https://github.com/CREATEGroup4/Challenge_Code/blob/main/CNN_accuracy.png">

> Train and validation loss per epoch for the CNN model

<img width="516" alt="image" src="https://github.com/CREATEGroup4/Challenge_Code/blob/main/CNN_loss.png">

> Train and validation Accuracy per epoch for the LSTM model

<img width="516" alt="image" src="https://github.com/CREATEGroup4/Challenge_Code/blob/main/LSTM_loss.png">

> Train and validation loss per epoch for the LSTM model

<img width="516" alt="image" src="https://github.com/CREATEGroup4/Challenge_Code/blob/main/LSTM_accuracy.png">

#### Sub-task 1: Tool detection
How did you come up with your solution for the subtask?
  - Based on the predicted length of time required to train the network (~2 days) we decided to focus on optimizing the training process as these were modifications we could make quickly such that enough time would be left to train the model.

Did you modify/optimize the training process in any way? 
  - We modified the way training/validation frames were selected. Since the datasets contained novice and experienced particpants we predicted that techniques would vary between these two subgroups. To maximize variation captured in the validation set, we selected 1 novice and 1 experienced particpant to be held out of the training set and to be used as the validation set. All frames from all videos these particpants were in were used for validation. 
  - After creating a histogram of the number of frames each tool was present in for the first particpant we realized the dataset was very imbalanced. To further enhance training we implemented code to create a more balanced distribution of frames containing each tool in each epoch. We did not have time to train the model with this modification.
  <img width="516" alt="image" src="https://user-images.githubusercontent.com/104586249/236253637-c358f02e-0b48-4fc6-82d9-c172085e30c9.png">

How did you decide to split the data for training?  
  - As described above, we split training and validation data at the participant level considering the 2 groups of particpants (experts and novicces).
Did you use any preprocessing or postprocessing? If so, what?  
  - We did not modify preprocessing or postprocessing during this challenge
How much time did you dedicate to this task (approximately)?  
  - We dedicated about half our time to this task.
  

#### Sub-task 2: Workflow recognition
How did you come up with your solution for the subtask?  
  - For this subtask, since the model runs faster than subtask 1 we began with training the base model with the same modifications made to training/validation split as subtask 1. We also turned on the "balance" flag such that frame balancing was implemented for each epoch. We did this because a histogram of the tasks for one of the videos demonstrated a large class imblance (figure shown below). The initial training curve was strange, so we adjusted the learning rate and saw enhanced performance. Due to the time limit we did not make further modifications to the network and used the trained network to predict labels for the unlabelled data.
  
  <img width="522" alt="image" src="https://user-images.githubusercontent.com/104586249/236267568-fa071ab7-48eb-48e2-8c10-b95bbc199ed6.png">

Did you modify/optimize the training process in any way?  
  - We performed the same divide as subtask 1 for training and vaidation split. We additionally implemented 8-fold cross validaiton. This allowed us to train the model 8 times (with different participants held out for validaiton in each fold). We used an enesemble of the results from each fold to produce our final results. This allowed to capture the variability of the data with minimal code edits.
  
How did you decide to split the data for training?  
  - Same as subtask 1
  
Did you use any preprocessing or postprocessing? If so, what?
  -  We did not modify preprocessing or postprocessing during this challenge

How much time did you dedicate to this task (approximately)?  
  - We dedicated time during the second week to this task.

#### Unlabelled Data
How did you approach using the unlabelled data?  
  - We used the best-trained model for each sub-task as of the beginning of week 2 to create an initial set of labels for the unlabelled data. Using the annotation reviewer we updated the labels we felt were incorrect before submitting the labels for review. Each group member annotated 5 videos. 
  
Did you use it during the training process?  
  - Once unlabelled videos were approved, we used them as a test set to determine how accurately our sub-task 2 model was working. This was helpful for tuning the model before the formal test set was released.
  
What tools did you choose to use to annotate/review the labels?  
  - We chose to use Rebeccas annotator reviewer from her GitHub because it was designed for the dataset and therefore made modifications easy.

How much time did you dedicate to this task (approximately)?  
  - We dedicated most of the second week to this task.

## Setup:
How did you set up your environment? Include detailed steps. If you used the environment that was provided, you may simply refer back to the original challenge page.  
  - please refer back to the original challenge page

If you used the provided environment, did you install any additional libraries? Include steps for installation (can be a link to existing documentation)
   - No additional libraries were used.

## Training the networks:
Note that your networks must take in the dataset using the same csv format as the baseline networks provided. No restructuring of the data folders is permitted.

#### Sub-task 1: Tool detection
Include steps to reproduce your trained network
The model for tool detection was trained using the GPU server and the methods from the CREATE Central Line Challenge baseline instructions:
```
conda activate createKerasEnv

python /path/to/Create_Group4_Challenge_Code/Tool_Detection/Train_Yolov3.py --save_location=/path/to/save_loc --data_csv_file=/path/to/Training_Data/Training_Data.csv
```
  
#### Sub-task 2: Task Recognition
> Include steps to reproduce your trained network 


To reproduce the results for task recognition, run the following code on the jovyan cluster:
```
cd ~/Documents/Mohamed-Train2
./cross_val.sh # train an ensemble of 8 models, each using a different fold for validation
```

## Generating test predictions
Note that your code must take in the dataset using the same csv format as the baseline networks provided and produce results that are correctly formatted for submission. No restructuring of the data folders is permitted.

#### Sub-task 1: Tool detection
Include steps on how to produce predictions on test results
Test predictions were generated using the GPU server and the methods from the CREATE Central Line Challenge baseline instructions:
```
python /path/to/Create_Group4_Challenge_Code/Tool_Detection/generateTestPredictions.py --save_location=/path/to/save_loc --data_csv_file=/path/to/Test_Data/Test_Data.csv
```
  
#### Sub-task 2: Task Recognition
> Include steps on how to produce predictions on test results.

To reproduce the results for task recognition, run the following code on the jovyan cluster:
```
cd ~/Documents
python3 Mohamed-Train2/pred_eval_1.py # run an ensemble model
python3 Mohamed-Train2/ensemble_preds.py # average the predictions to create the final result
```

## Final Results

#### Sub-task 1 Results:
mAP (IoU 50%): 9.685240495080606e-06
mAP (IoU 75%): 0.0
mAP (IoU 90%): 0.0

#### Sub-task 2 Results:
Class-weighted accuracy: 0.66
Average precision: 0.60
Average recall: 0.53
Starting transitional delay: 0.46
End transitional delay: 0.5

#### Overall Scores:
Sub-task 1 overall results: 9.685e-06
Sub-task 2 overall results: 0.83
Annotation score = 0.156
Overall score: 0.986

## Final reflections
Include a short reflection on your general strategy.  
  - Our strategy was to focus on fast modifications that had the potential to make large impacts in the accuracy of the models. We largely focused on balancing the datasets, and for subtask 2 which trained relatively quickly we also implemented 8-fold cross validation and adjusted the learning rate to enhance the quality of uro training and validation curves. This strategy allowed is to receive results early on in the competition, but limited the coding to be completed. 
 
Do you think that your strategy worked well?  
  - The strategy seemed to work well overall however, we did not have enough time to train the tool-recognition model with tool balancing. Because sub-task 1 model takes so long to train, al modifications to this network should have been made in 1 batch so we only had to train the model once with updated balancing.
  - Our strategy seemed to be most effective for sub-task 2 as the model trains faster to modifications to the training process can be tested quickly.
  
What would you do differently next time?  
  - Produce intitial labels for the unlabelled data earlier so we can spend more time submitting results for the unlabelled dataset. This was relativey fast to edit with the annotation reviewer and is a fast way to get some extra points.
  - In the first meeting, we would spend more time reviewing the skills everyone in the group has, and filling in gaps in each other's knowledge. Since there was a wide range of experience within our group, we tried to use everyone's abilities however this meant we did not take enough time to explain concepts to all group members. It is important that everyone feels they are contributing, learning, and on the same page.
  - Most things we would do differently are with our strategy for sub-task 1. Since the model took a signifcant amount of time to train the number of implementations we could train were limited. In the future, we would reduce the size of the training set for initial testing. This way we could try to get the model working by overfitting a small training set, implement balancing with the small dataset to make sure code works, and then we could perform a systematic review of hyperparamters. Then, we could switch to formally training our full dataset.

Do you have any advice for future participants of the challenge?  
  - Make sure to start training models early (espeically the model for sub-task 1 as each training run takes approximately 2 days)
  - It may be worth reducing the amount frames in the training data for sub-task 1 to enhance training speed, especially for initially testing ideas for enhancing network performance.
  - As soon as you can produce iniital labels for unlabelled data, review and submit the labels for extra points.
  
 
