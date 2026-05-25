# 데이터셋 위치

실제 데이터셋 파일은 Git에 올리지 않습니다.

추천 구조:

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

현재 프로젝트 데이터셋은 실제 class 이름을 아래처럼 사용합니다.

```text
datasets/processed/eye/
  train/
    awake/
    sleepy/
  val/
    awake/
    sleepy/
  test/
    awake/
    sleepy/

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

정리 스크립트:

```bash
cd train_model
python prepare_datasets.py
```

원본 데이터셋은 아래처럼 두는 것을 추천합니다.

```text
datasets/raw/
```

`raw/`, `processed/`는 `.gitignore`에 의해 제외됩니다.
