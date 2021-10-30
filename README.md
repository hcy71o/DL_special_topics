# DL Assignment
For submission

## Preparing & Model setting
1. Git clone
```
git clone -b master --single-branch https://github.com/hcyspeech/DL_special_topics.git
```
2. You can set model configuration of other hyperparameters in config/default.yaml file
(bn, dropout, dropout ratio, optimizer, lr, epoch, weight decay, ....)

## Training and test
[1번]
Train model with below codes

A, B, C mean each CNN structure of problem1 (A.Normal Conv, B.1x1 Conv, C.Deptwise Conv)

```
python main.py --t A
python main.py --t B
python main.py --t C
```

[2번] 
Train model with below codes (inside 2_GAN folder)
```
cd 2_GAN
python GAN.py
```
## Evaluation
In problem 1, model type (A,B,C) and configuration or hyperparmeters setting is converted to 6 hash values.

Each model experiment generates different folder.

In each folder, there are first json file which has experimental result and second, checkpoint file which save model parameters.

Accuracy and loss values can be visualized by extracting them from json file in each folder.

## TODO
Implement checkpoint option (optional)

