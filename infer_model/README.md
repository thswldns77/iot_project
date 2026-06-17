# Raspberry Pi Drowsiness Inference

This `infer_model` directory is the **Raspberry Pi inference package**. Copy this
directory to the Raspberry Pi and run it there.

Training happens on a personal PC or Google Colab. The Raspberry Pi only loads
the trained TFLite models and runs real-time inference.

```text
Personal PC
Dataset -> MobileNet fine-tuning -> TFLite model export

Raspberry Pi 5
NoIR/RGB camera -> MediaPipe -> TFLite inference -> drowsiness decision
-> ultrasonic head-drop assist -> servo alert and optional BLE phone alert
```

## Directory Structure

```text
infer_model/
  ble_alert.py
  camera_server_picamera2.py
  main.sh
  night_led.py
  run_inference.py
  ultrasonic_head.py
  requirements.txt
  README.md
  models/
    eye_state_model.tflite
    mouth_state_model.tflite
```

The model filenames are temporary project defaults.

```text
Eye state model: models/eye_state_model.tflite
Mouth/yawn model: models/mouth_state_model.tflite
```

If trained models are not ready yet, use `--rule-only` to test the full camera,
MediaPipe, drowsiness-decision, and servo pipeline without TFLite models.

## Raspberry Pi 5 + NoIR Camera Flow

Runtime flow:

```text
1. Read frames from the NoIR camera on Raspberry Pi 5
2. Detect face landmarks with MediaPipe FaceMesh
3. Crop eye and mouth regions from landmark coordinates
4. Send eye crops to eye_state_model.tflite
5. Send mouth crops to mouth_state_model.tflite
6. Accumulate eye closure and yawning states over time
7. Read HC-SR04 distance and compare it with the calibrated normal-face baseline
8. Change status to DROWSY when the weighted drowsiness score is high enough
9. Move the SG90 servo on GPIO 18 when status is DROWSY
10. Optionally notify an Android phone over BLE when status changes
11. Return the servo to 0 degrees when status is not DROWSY
12. Keep the IR/LED illuminator on only during configured night hours
```

The system does not classify drowsiness from a single frame. It accumulates
state over a short time window.

Default drowsiness conditions:

```text
Eye closed for 2.0 seconds or longer: +0.60
Mouth open/yawn for 3.0 seconds or longer: +0.55
Eye-closed ratio over the last 5 seconds is 45% or higher: +0.55
MediaPipe head-pose down, only with --enable-head: +0.50
Ultrasonic head-drop assist: +0.15

DROWSY when score >= 0.50
```

MediaPipe head-pose detection is disabled by default. Add `--enable-head` to
use it. The ultrasonic head-drop assist starts automatically and has a low
weight, so it cannot mark the driver as drowsy by itself.

## Component Roles

```text
NoIR camera
Captures the driver's face in real time.

MediaPipe
Extracts face landmarks and eye/mouth crop positions.

eye_state_model.tflite
Classifies an eye crop as open or closed.

mouth_state_model.tflite
Classifies a mouth crop as normal or yawn/open.

Time-window decision logic
Filters out momentary blinks and combines sustained visual signals.

SG90 servo
Moves as an alert output when status is DROWSY.

BLE alert
Advertises the Raspberry Pi as `DrowsyPi` and notifies an Android app when
the drowsiness status changes.

HC-SR04 ultrasonic sensor
Uses the distance between the sensor and the user's face/head as a low-weight
head-drop assist signal.
```

## Model Output Convention

Temporary class convention:

```text
Eye model:
0 = open
1 = closed

Mouth model:
0 = normal
1 = yawn/open
```

If your trained model uses the opposite class order, change the positive class
index at runtime:

```bash
--eye-closed-index 0
--mouth-yawn-index 0
```

## Raspberry Pi Setup

Keep the system Python unchanged. Use a Python 3.11 virtual environment for
this project.

Install Raspberry Pi system packages:

```bash
cd ~/iot_project/infer_model
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-gpiozero python3-lgpio
```

If Python 3.11 is already available:

```bash
python3.11 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install setuptools wheel
python -m pip install -r requirements.txt
```

If the system uses Python 3.13 and `python3.11` is not available from `apt`, use
`pyenv` to install Python 3.11 without replacing the system Python.

After installing Python 3.11 with `pyenv`:

```bash
cd ~/iot_project/infer_model
pyenv local 3.11.9
python --version

rm -rf .venv
python -m venv .venv
source .venv/bin/activate
python --version

python -m pip install setuptools wheel
python -m pip install -r requirements.txt
```

The two `python --version` checks should both show Python 3.11.x.

If a NumPy 2.x compatibility error appears while loading `tflite-runtime`, reset
NumPy and OpenCV inside `.venv`:

```bash
cd ~/iot_project/infer_model
source .venv/bin/activate
python -m pip uninstall -y numpy opencv-python tflite-runtime
python -m pip install -r requirements.txt
```

Then confirm that NumPy is still 1.x:

```bash
python -c "import numpy; print(numpy.__version__)"
python -c "from tflite_runtime.interpreter import Interpreter; print('tflite OK')"
```

## Camera Check

Check that the NoIR camera is detected:

```bash
rpicam-hello --list-cameras
rpicam-hello --timeout 3000
```

If a preview appears, the camera connection is working.

NoIR cameras do not automatically see in the dark. For night testing, use an IR
LED illuminator.

## Run Without Trained Models

Use this mode before TFLite models are ready:

```bash
cd ~/iot_project/infer_model
source .venv/bin/activate
python run_inference.py --source picamera2 --mirror --servo-pin 18 --rule-only
```

This mode uses MediaPipe-derived EAR and MAR rules instead of trained models.
Head-pose rules are used only when `--enable-head` is added.

```text
EAR: eye-closure ratio
MAR: mouth-opening ratio
Pitch delta: head-drop angle change, only with --enable-head
```

When `--enable-head` is used, look straight at the camera for the first two
seconds. The script uses that period as the head-pose baseline.

## Run With Trained Models

Copy models trained on the PC to:

```text
infer_model/models/eye_state_model.tflite
infer_model/models/mouth_state_model.tflite
```

Then run:

```bash
source .venv/bin/activate
python run_inference.py --source picamera2 --mirror --servo-pin 18
```

## Run With Split Python Environments

Use this mode when the system Python can import `picamera2`, but the Python
3.11 virtual environment is needed for `mediapipe` and `tflite-runtime`.

```text
Terminal 1: system Python 3.13
Picamera2 -> local MJPEG stream

Terminal 2: Python 3.11 .venv
MJPEG stream -> MediaPipe -> TFLite -> drowsiness decision
```

### Run Camera Server And Inference Together

Use `main.sh` when you want to start the Picamera2 MJPEG server and inference
code with one command.

```bash
cd ~/iot_project/infer_model
./main.sh
```

The script does this automatically:

```text
1. Start camera_server_picamera2.py with /usr/bin/python3
2. Wait for the local MJPEG stream to start
3. Start night_led.py with /usr/bin/python3
4. Activate .venv
5. Run run_inference.py with --source mjpeg and --mirror
```

Pass normal `run_inference.py` options after `./main.sh`:

```bash
./main.sh --servo-pin 18
./main.sh --servo-pin 18 --ble-alert
```

If both an RGB camera and a NoIR/IR camera are connected, choose the camera
index with `CAMERA_INDEX`:

```bash
CAMERA_INDEX=1 ./main.sh --servo-pin 18 --ble-alert
```

The script reuses an already running camera server on the same port. If it
starts the camera server itself, it stops that server when inference exits.

By default, `main.sh` controls an IR/LED illuminator on BCM GPIO 17. The LED is
on from 18:00 to 07:00 and off during daytime. When `main.sh` exits, the LED is
turned off.

Change the pin or night hours with environment variables:

```bash
NIGHT_LED_PIN=22 ./main.sh --servo-pin 18
NIGHT_LED_START_HOUR=19 NIGHT_LED_END_HOUR=6 ./main.sh --servo-pin 18
```

Disable night LED control:

```bash
NIGHT_LED=0 ./main.sh --servo-pin 18
```

If the LED wiring is active-low, use:

```bash
NIGHT_LED_ACTIVE_LOW=1 ./main.sh --servo-pin 18
```

Terminal 1, without activating `.venv`:

```bash
cd ~/iot_project/infer_model
/usr/bin/python3 camera_server_picamera2.py --host 127.0.0.1 --port 8000 --width 640 --height 480 --fps 30
```

Keep Terminal 1 running. Then open another terminal.

If both an RGB camera and a NoIR/IR camera are connected, list the camera
indices first:

```bash
cd ~/iot_project/infer_model
/usr/bin/python3 camera_server_picamera2.py --list-cameras
```

Then start the MJPEG server with the RGB camera index:

```bash
/usr/bin/python3 camera_server_picamera2.py --camera-index 0 --host 127.0.0.1 --port 8000 --width 640 --height 480 --fps 30
```

If index `0` is the NoIR/IR camera, use `--camera-index 1` instead.

Terminal 2, with the Python 3.11 virtual environment:

```bash
cd ~/iot_project/infer_model
source .venv/bin/activate
python run_inference.py --source mjpeg --stream-url http://127.0.0.1:8000/stream.mjpg --mirror --servo-pin 18
```

To test without the servo:

```bash
python run_inference.py --source mjpeg --stream-url http://127.0.0.1:8000/stream.mjpg --mirror
```

The display shows:

```text
Status: AWAKE / DROWSY / NO FACE / CALIBRATING
Eye prob: probability of eye closed
Mouth prob: probability of yawn/open mouth
Pitch delta: head angle change from baseline, only active with --enable-head
Ultrasonic: distance, baseline, distance delta, and low-weight head-drop state
PERCLOS-ish: eye-closed ratio over the recent time window
```

## Servo Behavior

The default setup expects an SG90-style servo connected to BCM GPIO 18.

Run:

```bash
python run_inference.py --source picamera2 --mirror --servo-pin 18
```

When status is `DROWSY`, the servo repeats:

```text
0 degrees -> 90 degrees -> 0 degrees -> -90 degrees
```

When status is not `DROWSY`, the servo returns to 0 degrees.

To make the servo move faster:

```bash
python run_inference.py --source picamera2 --mirror --servo-pin 18 --servo-step-sec 0.5
```

Wiring notes:

```text
BCM GPIO 18 = physical pin 12
Use external 5V power if the servo is unstable
Connect Raspberry Pi GND and external power GND together
```

## Ultrasonic Head-Drop Assist

The HC-SR04 ultrasonic sensor is used as a low-weight helper signal. It does not
decide drowsiness alone.

Default behavior:

```text
Trigger pin: BCM GPIO 23, physical pin 16
Echo pin: BCM GPIO 24, physical pin 18
Baseline time: 3.0 seconds
Head-drop distance increase: 12 cm
Hold time: 1.0 second
Drowsiness score weight: +0.15
```

Wiring:

```text
HC-SR04 VCC  -> Raspberry Pi 5V
HC-SR04 GND  -> Raspberry Pi GND
HC-SR04 TRIG -> BCM GPIO 23
HC-SR04 ECHO -> voltage divider -> BCM GPIO 24
```

The HC-SR04 Echo output is usually 5V, but Raspberry Pi GPIO input is 3.3V.
Use a voltage divider before connecting Echo to GPIO 24:

```text
HC-SR04 ECHO ---- 1k ohm ---- GPIO24
GPIO24 ---------- 2k ohm ---- GND
```

Test only the ultrasonic sensor:

```bash
cd ~/iot_project/infer_model
source .venv/bin/activate
python ultrasonic_head.py
```

At startup, sit in the normal position for about 3 seconds. The detector stores
that distance as the baseline. If the measured distance increases by about 12 cm
and stays there for at least 1 second, `head_down=True` is reported.

If this error appears:

```text
Cannot determine SOC peripheral base address
```

remove old `RPi.GPIO` and use `rpi-lgpio` in the virtual environment:

```bash
python -m pip uninstall -y RPi.GPIO
python -m pip install -r requirements.txt
```

## BLE Phone Alert

Enable this when an Android app should receive drowsiness status changes over
Bluetooth Low Energy.

BLE convention:

```text
Device name: DrowsyPi
Service UUID: 0000d001-0000-1000-8000-00805f9b34fb
Characteristic UUID: 0000d002-0000-1000-8000-00805f9b34fb

Characteristic value:
0 = AWAKE
1 = DROWSY
2 = NO FACE or CALIBRATING
```

Check Bluetooth on Raspberry Pi:

```bash
systemctl status bluetooth
bluetoothctl
```

Inside `bluetoothctl`:

```text
show
power on
quit
```

Test BLE without camera inference:

```bash
cd ~/iot_project/infer_model
source .venv/bin/activate
python ble_alert.py
```

The test server advertises as `DrowsyPi` and cycles through `AWAKE`, `DROWSY`,
and `NO FACE`. Use the Android app or a BLE scanner app such as nRF Connect to
confirm that the service and characteristic are visible.

Run inference with BLE enabled:

```bash
python run_inference.py --source mjpeg --stream-url http://127.0.0.1:8000/stream.mjpg --mirror --servo-pin 18 --ble-alert
```

If the Android app uses different UUIDs, pass them at runtime:

```bash
python run_inference.py --source mjpeg --stream-url http://127.0.0.1:8000/stream.mjpg --mirror --ble-alert \
  --ble-device-name DrowsyPi \
  --ble-service-uuid 0000d001-0000-1000-8000-00805f9b34fb \
  --ble-characteristic-uuid 0000d002-0000-1000-8000-00805f9b34fb
```

## Tuning Options

Raise thresholds if the eye or mouth model is too sensitive:

```bash
python run_inference.py --source picamera2 --mirror --servo-pin 18 \
  --eye-threshold 0.9 \
  --mouth-threshold 0.9
```

Head-drop detection is disabled by default. Enable it only if you want to use
head pitch as an additional signal:

```bash
python run_inference.py --source picamera2 --mirror --servo-pin 18 \
  --enable-head \
  --head-drop-deg 20
```

Make the drowsiness decision stricter:

```bash
python run_inference.py --source picamera2 --mirror --servo-pin 18 \
  --eye-sec 2.5 \
  --yawn-sec 3.5 \
  --perclos-threshold 0.6
```

Run without a GUI window and print logs instead:

```bash
python run_inference.py --source picamera2 --mirror --servo-pin 18 --no-display
```

## Relationship With Training Code

This folder is inference-only. Training is done on a personal PC or Colab.

```text
train_model/
  train_classifier.py
  export_tflite.py

infer_model/
  run_inference.py
  models/
    eye_state_model.tflite
    mouth_state_model.tflite
```

PC workflow:

```text
1. Fine-tune MobileNetV3-Small on eye datasets
2. Fine-tune MobileNetV3-Small on mouth/yawn datasets
3. Export both models to TFLite
4. Copy the .tflite files into infer_model/models/
```

Raspberry Pi workflow:

```text
1. Read NoIR camera frames
2. Extract landmarks with MediaPipe
3. Run TFLite models for eye/mouth state
4. Combine time-window conditions and ultrasonic low-weight distance signal
5. Move the servo and optionally notify Android over BLE when status is DROWSY
```

## Quick Commands

Test without trained models:

```bash
cd ~/iot_project/infer_model
source .venv/bin/activate
python run_inference.py --source picamera2 --mirror --servo-pin 18 --rule-only
```

Run with trained models:

```bash
python run_inference.py --source picamera2 --mirror --servo-pin 18
```

Run with trained models and Android BLE alert:

```bash
python run_inference.py --source mjpeg --stream-url http://127.0.0.1:8000/stream.mjpg --mirror --servo-pin 18 --ble-alert
```
