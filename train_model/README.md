# 졸음 탐지 모델 훈련 코드

이 폴더는 개인 PC 또는 Google Colab에서 실행하는 **훈련 전용 코드**입니다.

라즈베리파이에서는 학습을 하지 않고, 여기서 만든 TFLite 모델만 복사해서 사용합니다.

```text
train_model
데이터셋 정리 -> MobileNetV3-Small fine-tuning -> TFLite 변환

infer_model
NoIR 카메라 -> MediaPipe landmark -> TFLite 추론 -> 졸음 판단 -> 서보 모터 작동
```

## 폴더 구조

```text
train_model/
  README.md
  requirements.txt
  train_classifier.py
  export_tflite.py
  datasets/
    README.md
```

## 파일별 역할

```text
train_classifier.py
눈/입 이미지 데이터셋을 읽어서 MobileNetV3-Small을 fine-tuning하는 메인 훈련 코드입니다.
학습 결과로 best.keras를 저장하고, --export-tflite 옵션을 주면 .tflite 모델도 바로 생성합니다.

export_tflite.py
이미 학습된 .keras 모델을 라즈베리파이 추론용 .tflite 모델로 변환하는 코드입니다.
학습은 하지 않고, float32 / float16 / int8 같은 변환 옵션을 실험할 때 사용합니다.

requirements.txt
개인 PC 또는 Colab에서 훈련에 필요한 Python 패키지를 정의합니다.

datasets/README.md
실제 데이터셋을 어떤 폴더 구조로 정리해야 하는지 설명합니다.
```

훈련 결과는 기본적으로 아래에 저장됩니다.

```text
train_model/outputs/
```

`outputs/`와 실제 데이터셋 폴더는 Git에 올리지 않습니다.

## 모델 구조

눈 모델과 입 모델을 따로 훈련합니다.

```text
눈 모델:
눈 crop 이미지 -> MobileNetV3-Small -> open / closed

입 모델:
입 crop 또는 얼굴 하단 crop 이미지 -> MobileNetV3-Small -> normal / yawn
```

라즈베리파이 추론 코드의 임시 모델 이름과 class 순서는 아래 기준입니다.

```text
infer_model/models/eye_state_model.tflite
0 = open
1 = closed

infer_model/models/mouth_state_model.tflite
0 = normal
1 = yawn/open
```

그래서 훈련할 때도 class 순서를 맞추는 것이 중요합니다.

## 데이터셋 구조

가장 추천하는 구조는 train/val/test를 직접 나누는 방식입니다.

```text
datasets/processed/eye/
  train/
    open/
    closed/
  val/
    open/
    closed/
  test/
    open/
    closed/

datasets/processed/mouth/
  train/
    normal/
    yawn/
  val/
    normal/
    yawn/
  test/
    normal/
    yawn/
```

`val/`이 없으면 `train/`에서 자동으로 validation split을 만듭니다.

데이터가 아직 나뉘어 있지 않다면 아래처럼 class 폴더만 있어도 됩니다.

```text
datasets/processed/eye/
  open/
  closed/
```

이 경우 `--validation-split` 값으로 train/validation을 나눕니다.

## 설치

개인 PC 또는 Colab에서 실행합니다.

권장 Python 버전은 3.11 또는 3.12입니다. TensorFlow와 TFLite 변환 호환성을 위해 Python 3.13보다는 3.11/3.12 가상환경을 추천합니다.

```bash
cd train_model
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Colab에서는 가상환경 없이 다음만 실행해도 됩니다.

```bash
pip install -r requirements.txt
```

## 눈 감김 모델 훈련

```bash
python train_classifier.py \
  --task eye \
  --data-dir datasets/processed/eye \
  --class-names open closed \
  --image-size 96 \
  --batch-size 32 \
  --head-epochs 10 \
  --fine-tune-epochs 10 \
  --export-tflite
```

결과 예시:

```text
outputs/eye/20260518-224500/best.keras
outputs/eye/20260518-224500/eye_state_model.tflite
outputs/eye/20260518-224500/metadata.json
```

생성된 `eye_state_model.tflite`를 라즈베리파이 repo의 아래 위치에 복사합니다.

```text
infer_model/models/eye_state_model.tflite
```

## 입/하품 모델 훈련

```bash
python train_classifier.py \
  --task mouth \
  --data-dir datasets/processed/mouth \
  --class-names normal yawn \
  --image-size 128 \
  --batch-size 32 \
  --head-epochs 10 \
  --fine-tune-epochs 10 \
  --export-tflite
```

생성된 `mouth_state_model.tflite`를 라즈베리파이 repo의 아래 위치에 복사합니다.

```text
infer_model/models/mouth_state_model.tflite
```

## 훈련 방식

스크립트는 두 단계로 학습합니다.

```text
1단계:
ImageNet으로 사전학습된 MobileNetV3-Small backbone을 freeze
마지막 분류층만 학습

2단계:
MobileNetV3-Small의 마지막 일부 layer만 unfreeze
작은 learning rate로 fine-tuning
```

기본값:

```text
head learning rate: 1e-3
fine-tuning learning rate: 1e-5
fine-tuning layers: 마지막 40개 layer
```

데이터가 적으면 `--fine-tune-layers`를 줄이는 편이 안전합니다.

```bash
--fine-tune-layers 20
```

## TFLite 변환만 따로 하기

이미 저장된 `.keras` 모델을 다시 TFLite로 변환할 수 있습니다.

```bash
python export_tflite.py \
  --keras-model outputs/eye/20260518-224500/best.keras \
  --output outputs/eye/20260518-224500/eye_state_model.tflite \
  --quantization float32
```

라즈베리파이에서 속도가 부족하면 float16도 시도할 수 있습니다.

```bash
python export_tflite.py \
  --keras-model outputs/eye/20260518-224500/best.keras \
  --output outputs/eye/20260518-224500/eye_state_model_float16.tflite \
  --quantization float16
```

int8 변환은 representative dataset이 필요합니다.

```bash
python export_tflite.py \
  --keras-model outputs/eye/20260518-224500/best.keras \
  --output outputs/eye/20260518-224500/eye_state_model_int8.tflite \
  --quantization int8 \
  --representative-dir datasets/processed/eye/train \
  --image-size 96
```

처음에는 `float32` TFLite로 동작 확인 후, 속도가 부족할 때 `float16` 또는 `int8`로 넘어가는 것을 추천합니다.

## 라즈베리파이 추론 코드와의 연결

이 훈련 코드는 TFLite 모델 입력을 기본적으로 `0~1` 범위로 맞춥니다.

따라서 라즈베리파이에서는 기본 실행 그대로 사용하면 됩니다.

```bash
python run_inference.py --source picamera2 --mirror --servo-pin 18
```

별도로 `--input-normalization zero_one`을 주지 않아도 기본값이 `zero_one`입니다.

## 주의할 점

이미지 단위로 랜덤 split하면 같은 사람의 비슷한 프레임이 train/test에 동시에 들어갈 수 있습니다.

가능하면 데이터셋을 나눌 때는:

```text
사람 기준 분리
영상 기준 분리
촬영 세션 기준 분리
```

를 우선하는 것이 좋습니다.

특히 졸음 탐지는 실사용 환경 차이가 큽니다. 공개 데이터셋으로 먼저 학습한 뒤, 라즈베리파이5 + NoIR 카메라로 직접 찍은 샘플을 조금 추가해서 fine-tuning하면 성능이 더 안정적입니다.
