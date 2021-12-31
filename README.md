

# Explanation of hyperparameters

1. batch size
    * meaning : Number of data included in a unit of computation.
    * data type : single integer
    * default value : 256


2. epoch
    * meaning : Number of complete training.
    * data type : single integer
    * default value : 100


3. learning rate
    * meaning : Variables that determine the step of movement in the process of loss function.
    * data type : single real number
    * default value : 0.001


4. weight decay
    * meaning : Variables used to control the weight of the model in loss function.
    * data type : single real number
    * default value : 0.0001


# command to train model
python main.py train {데이터셋파일이름} {모델이름}

# command to test model
python main.py test {데이터셋파일이름} {모델이름}

# path to your data
