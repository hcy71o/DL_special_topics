# DL Assignment
This is for assignment
## Prepare dataset
Download CIFAR-10 dataset from the internet by running:
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz
```

## Training

Training process can be ran by:
```
python train.py --config config_vp.json 
```
## Inference
To generate samples, run the following command:
```
python syn.py --step 00900000 
```
Feel free to change the step as desired. 
