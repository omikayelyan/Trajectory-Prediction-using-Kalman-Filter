#  Real-Time Object Tracking & Trajectory Prediction using Kalman Filter

This project demonstrates real-time object tracking and trajectory prediction using a Kalman Filter and a webcam.

The system detects a colored object (default: pink), tracks its movement, filters noise, and predicts future positions in real time.

---

##  Features

* Real-time webcam tracking (OpenCV)
* Color-based object detection (HSV filtering)
* Kalman filter smoothing (noise reduction)
* Future trajectory prediction
* Prediction accuracy evaluation (live error calculation)
* Average prediction error displayed in real time

---

## How It Works

The system uses a **Kalman Filter** with a constant velocity model.

### State Vector:

```
[x, y, vx, vy]
```

### Process:

1. Detect object position from camera
2. Predict next state using motion model
3. Update prediction with measurement
4. Estimate future trajectory
5. Compare prediction with actual next position

---

##  Demo Visualization

*  Green → Measured position (from camera)
*  Blue → Kalman filtered position
*  Red → Future predicted trajectory
*  Yellow → Previous frame prediction vs actual (error visualization)

---

##  Installation

Make sure you have Python 3 installed.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python kalman_tracker.py
```

---

## Controls

* `q` → Quit
* `c` → Reset tracking

---

## Testing the System

Use a **bright green object** (marker, ball, etc.).

Try:

* Straight motion → low prediction error
* Smooth curves → moderate error
* Sudden direction change → higher error (expected)

---

## Evaluation Method

Prediction accuracy is evaluated using Euclidean distance:

```
error = sqrt((x_real - x_pred)^2 + (y_real - y_pred)^2)
```

The system displays:

* Last prediction error
* Average prediction error
* Number of evaluated frames

---

##  Possible Improvements

* Add acceleration model (Extended Kalman Filter)
* Use object detection (YOLO instead of color)
* Record trajectory data to file
* Track multiple objects

---

## Technologies Used

* Python
* OpenCV
* NumPy

---
