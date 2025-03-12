# Challenge_Code
This repository should include all materials needed to reproduce your results file. Please fill in the following sections to describe your approach to the challenge and how to set up your environment and run your code. Minimum questions to be answered are listed in each section, but feel free to add more details beyond what's listed. The information included in this file, is also what is expected in your final results presentation.  
  
This ReadME file must be complete by the results presentation on May 8th, 2024 to be eligible for awards.    
  
Winning teams (for each subtask and overall) will have their submission featured in the Winning_Strategies repository on the main challenge page for all future participants to see.  
  
## Team members:
Maha Kesibi, Sarah Nassar, Dharsan Ravindran, Kai La (Jennifer) Tsang, Paul Wilson (mentor)
  
## Strategy:
#### General approach
- **How did you approach the challenge overall? Did you target one particular subtask over the others?** We aimed to work on each sub-task.
- **How did you divide the work to be done amongst all team members?** Each team member got to choose to work on their sub-task of interest. Maha and Jennifer worked on sub-task 1, Sarah and Dharsan worked on sub-task 2, and Paul worked on sub-task 3.
- **How did you make use of the resources provided (e.g. base-line code)** We used the baseline code to check the preliminary performance, then brainstormed ideas to improve. For sub-task 2, we used the GPU server that was provided for the challenge.

#### Sub-task 1: Tool detection
- **What was your general strategy for approaching this subtask?**
- **How did you come up with your solution for the subtask?**
- **Did you modify/optimize the training process in any way?**
- **How did you decide to split the data for training?**
- **Did you use any preprocessing or postprocessing? If so, what?**
- **How much time did you dedicate to this task (approximately)?**

#### Sub-task 2: Workflow recognition
- **What was your general strategy for approaching this subtask?** Run the baseline then try to understand the code and look for ways to improve (parameter changes and/or network changes)
- **How did you come up with your solution for the subtask?** After noticing that the TCN/LSTM did not help improve the performance, the solution only included the ResNet50 CNN with the default baseline code settings
- **Did you modify/optimize the training process in any way?** Turn on augmentation and balancing, increase/decrease the sequence length, try TCN instead of LSTM, decrease the batch size, increase the learning rate, increase the patience for early stopping, reduce the frequency of learning rate updates, try InceptionV3 instead of ResNet50
- **How did you decide to split the data for training?** AN03 and MS04 data was used for validation, everyone else's data was used for training
- **Did you use any preprocessing or postprocessing? If so, what?** Turn on augmentation, explore test set images and predictions
- **How much time did you dedicate to this task (approximately)?** A few days

#### Sub-task 3: Ultrasound segmentation
- **What was your general strategy for approaching this subtask?**
- **Which foundation model did you use and why?**
- **Did you fine-tune the model? If so, how did you do this?**
- **Did you make use of the existing labels? If so, how did you incorporate them?**
- **Did you use any preprocessing or postprocessing? If so, what?**
- **How much time did you dedicate to this task (approximately)?**

#### Unlabelled Data
- **How did you approach using the unlabelled data?**
- **Did you use it during the training process?**
- **What tools did you choose to use to annotate/review the labels?**
- **How much time did you dedicate to this task (approximately)?**

## Setup:
How did you set up your environment? Include detailed steps. If you used the environment that was provided, you may simply refer back to the original challenge page.  
If you used the provided environment, did you install any additional libraries? Include steps for installation (can be a link to existing documentation)

## Training the networks:
Note that your networks must take in the dataset using the same csv format as the baseline networks provided. No restructuring of the data folders is permitted.
#### Sub-task 1: Tool detection
Include steps to reproduce your trained network
#### Sub-task 2: Task Recognition
Include steps to reproduce your trained network
```
conda activate createPytorchEnv
python /.../TaskRecognition/Train_CNN.py --save_location=/.../TaskRecognition --data_csv_file=/.../Training_Data/Training_Data.csv
```
#### Sub-task 3: Ultrasound probe segmentation
Include steps to reproduce your trained network

## Generating test predictions
Note that your code must take in the dataset using the same csv format as the baseline networks provided and produce results that are correctly formatted for submission. No restructuring of the data folders is permitted.
#### Sub-task 1: Tool detection
Include steps on how to produce predictions on test results
#### Sub-task 2: Task Recognition
Include steps on how to produce predictions on test results
```
conda activate createPytorchEnv
python /.../TaskRecognition/generateTestPredictions.py --save_location=/.../TaskRecognition --data_csv_file=/.../Test_Data/Test_Data.csv
```
#### Sub-task 3: Task Recognition
Include steps on how to produce predictions on test results

## Final reflections
- **Include a short reflection on your general strategy.**
- **Do you think that your strategy worked well?**
- **What would you do differently next time?**
- **Do you have any advice for future participants of the challenge?**
