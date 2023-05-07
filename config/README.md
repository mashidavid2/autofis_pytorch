# Recommendation

|             |                          |
| ----------- | ------------------------ |
| contributor | seunghyun_yoo            |
| email       | seunghyun_yoo@tmax.co.kr |

## Step 1. Data Preparation

### 1. Data object table에서 사용할 data object 선택

### 2. 학습에 필요한 data 정보 입력

```
- (required) userId, itemId column는 필수로 입력 해야 하는 정보
- (optional) user가 item을 얼마나 선호하는 지 학습 할 때 평점(rating) 등의 정확한 지표를 사용해 학습하고자 하는 경우 rating column 선택
- (optional) user, item features를 활용 하고자 하는 경우 user, item feature column선택
```

## Step 2. Model Selection

### 1. 모델 선택

```
- 제공하는 모델 : Matrix Factorization, Factorization Machine
```

### 2. (advanced option) model finetuning을 원할 경우, advanced option을 통해 hyperparameter 설정

### 3. 실험 생성
