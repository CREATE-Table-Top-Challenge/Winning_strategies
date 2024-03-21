# Results
This repository should include your final results submission. Predictions on test data must be submitted as a single csv file.  
Your CSV file must have the following columns (exactly):
1. FileName
> - 1 row per file
2. Overall Task
> - 1 entry per file
> - class names must be the same as those provided in the training data
3. Tool bounding box
> - 1 entry per file
> - format: each entry should be a list of dictionaries. 1 dictionary per predicted bounding box.
>> - dictionaries should have the format: {"class":str, "xmin":int, "xmax":int, "ymin":int, "ymax":int, "conf":float}
  
#### Predictions must be generated from code. To be considered for awards, no manual post-processing is permitted. Organizers will be checking that results generated from your submitted code match those in the submitted csv file before results are made available to the public. 
