# 라즈베리파이 졸음 탐지 추론 코드

이 `infer_model` 폴더는 라즈베리파이에 그대로 복사해서 사용하는 **추론 전용 폴더**입니다.

개인 PC에서는 모델을 훈련하고, 라즈베리파이에서는 훈련된 모델을 이용해 실시간으로 졸음 여부를 판단합니다.

```text
개인 PC
데이터셋 -> MobileNet 계열 모델 훈련 -> TFLite 모델 생성

라즈베리파이5
NoIR 카메라 -> MediaPipe -> TFLite 추론 -> 졸음 판단 -> 서보 모터 작동
```

## 전체 구조

```text
infer_model/
  run_inference.py
  requirements.txt
  README.md
  models/
    eye_state_model.tflite
    mouth_state_model.tflite
```

현재 모델 파일명은 임시로 정해두었습니다.

```text
눈 감김 모델: models/eye_state_model.tflite
하품/입 벌림 모델: models/mouth_state_model.tflite
```

아직 훈련된 모델이 없으면 `--rule-only` 옵션으로 모델 없이 전체 흐름을 먼저 테스트할 수 있습니다.

## 라즈베리파이5 + NoIR 카메라 전체 플로우

실제 실행 흐름은 아래와 같습니다.

```text
1. Raspberry Pi 5에서 NoIR 카메라 프레임을 읽음
2. MediaPipe FaceMesh가 얼굴 landmark를 추출함
3. landmark 좌표로 눈 영역과 입 영역을 crop함
4. 눈 crop을 eye_state_model.tflite에 입력함
5. 입 crop을 mouth_state_model.tflite에 입력함
6. MediaPipe landmark로 고개 pitch 각도를 계산함
7. 최근 몇 초 동안의 눈 감김, 하품, 고개 떨굼 상태를 누적함
8. 조건을 넘으면 상태를 DROWSY로 바꿈
9. DROWSY 상태일 때 GPIO 18번의 SG90 서보 모터를 움직임
10. 졸음 상태가 아니면 서보를 0도로 되돌림
```

즉, 라즈베리파이에서 한 프레임마다 바로 졸음이라고 판단하는 것이 아니라, **최근 몇 초 동안의 상태를 누적해서 판단**합니다.

기본 졸음 판단 조건은 다음과 같습니다.

```text
눈 감김이 1.0초 이상 지속됨
또는 하품/입 벌림이 2.0초 이상 지속됨
또는 고개 떨굼이 1.5초 이상 지속됨
또는 최근 5초 동안 눈 감김 비율이 45% 이상
```

## 각 구성 요소의 역할

```text
NoIR 카메라
운전자 얼굴 영상을 실시간으로 입력받음

MediaPipe
얼굴 landmark를 추출해서 눈/입 crop 위치와 고개 각도를 계산함

eye_state_model.tflite
눈 crop을 보고 open / closed를 분류함

mouth_state_model.tflite
입 crop을 보고 normal / yawn 또는 mouth open을 분류함

시간 기반 판단 로직
순간적인 깜빡임과 실제 졸음을 구분하기 위해 최근 몇 초간의 결과를 누적함

SG90 서보 모터
DROWSY 상태일 때 경고 동작으로 움직임
```

## 모델 출력 규칙

임시 규칙은 아래와 같습니다.

```text
눈 모델 출력:
0 = open
1 = closed

입 모델 출력:
0 = normal
1 = yawn/open
```

훈련한 모델의 class 순서가 반대라면 실행할 때 옵션을 바꾸면 됩니다.

```bash
--eye-closed-index 0
--mouth-yawn-index 0
```

## 라즈베리파이 설치

시스템 Python은 바꾸지 말고, 이 프로젝트용 가상환경만 Python 3.11로 만드는 것을 추천합니다.

```bash
cd ~/junhyuk/infer_model
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-gpiozero python3-lgpio
```

Python 3.11이 있다면:

```bash
python3.11 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Python 3.11이 없다면, 시스템 Python을 교체하지 말고 Python 3.11만 추가 설치합니다.

```bash
sudo apt install -y python3.11 python3.11-venv
```

그 다음 다시 가상환경을 만듭니다.

```bash
python3.11 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`--system-site-packages`를 쓰는 이유는 `picamera2`, `opencv`, `gpiozero` 같은 라즈베리파이 패키지를 `apt`로 설치했기 때문입니다.

## 카메라 확인

먼저 NoIR 카메라가 잡히는지 확인합니다.

```bash
rpicam-hello --list-cameras
rpicam-hello --timeout 3000
```

화면이 뜨면 카메라 연결은 정상입니다.

NoIR 카메라는 어두운 곳에서 자동으로 밝게 보이는 카메라가 아닙니다. 야간 환경에서는 IR LED 같은 적외선 조명이 필요합니다.

## 모델 없이 먼저 실행

아직 훈련된 TFLite 모델이 없다면 아래 명령으로 전체 파이프라인을 먼저 확인합니다.

```bash
cd ~/junhyuk/infer_model
source .venv/bin/activate
python run_inference.py --source picamera2 --mirror --servo-pin 18 --rule-only
```

이 모드는 모델 대신 MediaPipe에서 계산한 EAR, MAR, head pose 값을 이용합니다.

```text
EAR: 눈 감김 정도
MAR: 입 벌림 정도
Pitch delta: 고개 숙임 정도
```

처음 2초 동안은 정면을 보고 있어야 합니다. 이때의 얼굴 각도를 기준 자세로 저장합니다.

## 훈련된 모델로 실행

PC에서 훈련한 모델을 아래 경로에 복사합니다.

```text
infer_model/models/eye_state_model.tflite
infer_model/models/mouth_state_model.tflite
```

그 다음 실행합니다.

```bash
source .venv/bin/activate
python run_inference.py --source picamera2 --mirror --servo-pin 18
```

화면에는 다음 값들이 표시됩니다.

```text
Status: AWAKE / DROWSY / NO FACE / CALIBRATING
Eye prob: 눈 감김 확률
Mouth prob: 하품 또는 입 벌림 확률
Pitch delta: 기준 자세 대비 고개 각도 변화
PERCLOS-ish: 최근 몇 초 동안 눈 감김 비율
```

## 서보 모터 동작

기본 설정은 SG90 서보 모터를 GPIO 18번, 즉 BCM 18에 연결하는 방식입니다.

실행 명령:

```bash
python run_inference.py --source picamera2 --mirror --servo-pin 18
```

`DROWSY` 상태가 되면 서보가 다음 순서로 반복해서 움직입니다.

```text
0도 -> 90도 -> 0도 -> -90도
```

졸음 상태가 아니면 0도로 돌아갑니다.

더 빠르게 움직이게 하려면:

```bash
python run_inference.py --source picamera2 --mirror --servo-pin 18 --servo-step-sec 0.5
```

주의:

```text
BCM GPIO 18 = 물리 핀 12번
서보 전원이 불안정하면 외부 5V 전원을 사용
라즈베리파이 GND와 외부 전원 GND는 반드시 공통 연결
```

## 주요 튜닝 옵션

눈 감김이나 하품 모델이 너무 예민하면 threshold를 올립니다.

```bash
python run_inference.py --source picamera2 --mirror --servo-pin 18 \
  --eye-threshold 0.7 \
  --mouth-threshold 0.7
```

고개 떨굼이 너무 예민하면 각도 기준을 올립니다.

```bash
python run_inference.py --source picamera2 --mirror --servo-pin 18 \
  --head-drop-deg 20
```

졸음 판단 시간을 더 엄격하게 하려면:

```bash
python run_inference.py --source picamera2 --mirror --servo-pin 18 \
  --eye-sec 1.5 \
  --yawn-sec 3.0 \
  --head-sec 2.0
```

화면 없이 SSH 로그만 보고 싶으면:

```bash
python run_inference.py --source picamera2 --mirror --servo-pin 18 --no-display
```

## 학습 코드와의 관계

이 폴더는 추론 전용입니다. 훈련은 개인 PC나 Colab에서 합니다.

```text
training/
  train_eye_model.py
  train_mouth_model.py
  export_tflite.py

infer_model/
  run_inference.py
  models/
    eye_state_model.tflite
    mouth_state_model.tflite
```

PC에서 할 일:

```text
1. 눈 데이터셋으로 MobileNetV3-Small fine-tuning
2. 입/하품 데이터셋으로 MobileNetV3-Small fine-tuning
3. TFLite로 변환
4. 생성된 .tflite 파일을 infer_model/models/에 복사
```

라즈베리파이에서 할 일:

```text
1. NoIR 카메라 입력
2. MediaPipe로 landmark 추출
3. TFLite 모델로 눈/입 상태 추론
4. 고개 각도와 시간 조건을 함께 판단
5. DROWSY면 서보 모터 작동
```

## 빠른 실행 요약

모델 없이 테스트:

```bash
cd ~/junhyuk/infer_model
source .venv/bin/activate
python run_inference.py --source picamera2 --mirror --servo-pin 18 --rule-only
```

모델 넣고 실행:

```bash
python run_inference.py --source picamera2 --mirror --servo-pin 18
```
