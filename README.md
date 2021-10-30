# DL Assignment
For submission

## Preparing & Model setting
1. Git clone
```
git clone -b master --single-branch https://github.com/hcyspeech/DL_special_topics.git
```
2. config/default.yaml 파일에서 모델 구조와 훈련 관련 하이퍼파라미터 설정가능
(bn, dropout, dropout ratio, optimizer, lr, epoch, weight decay, ....)

## Training and test
[1번]
아래 코드로 training 실행 

A, B, C는 1번 문제의 각각 다른 CNN구조 의미 (각각 A.Normal Conv, B.1x1 Conv, C.Deptwise Conv)

```
python main.py --t A
python main.py --t B
python main.py --t C
```

[2번] 
아래 코드로 training 실행 (2_GAN 폴더 내부에 존재)
```
cd 2_GAN
python main.py --t A
```
## Evaluation
1번에서, 모델 종류 (A, B, C)와 yaml 파일 설정 값에 따른 각각 다른 모델 구조는 6자리 해쉬 값으로 전환되어 각 폴더에 저장

폴더 내부에는 실험결과가 내장 된 json 파일과 파라미터가 저장된 체크포인트 파일 두 가지 파일이 존재

train/test accuracy plot이나 최종 test accuracy 등은 json파일에서 불러와서 시각화

## TODO
체크포인트 기능 구현 (optional)

