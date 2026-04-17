import cv2
import numpy as np
from collections import deque
import math

"""
Real-time object trajectory prediction with a Kalman filter.

How it works:
- Detects a bright PINK object from the camera.
- Green = measured position
- Blue = Kalman filtered position
- Red = future predicted trajectory
- Yellow = previous frame's prediction for the current frame

Controls:
- Press 'q' to quit
- Press 'c' to clear/reset
"""

WARMUP_DETECTIONS = 3


def nothing(_):
    pass


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def create_hsv_trackbars():
    cv2.namedWindow("Controls")
    cv2.resizeWindow("Controls", 420, 320)

    # Default range for green objects in HSV
    cv2.createTrackbar("H Low", "Controls", 140, 179, nothing)
    cv2.createTrackbar("S Low", "Controls", 60, 255, nothing)
    cv2.createTrackbar("V Low", "Controls", 80, 255, nothing)
    cv2.createTrackbar("H High", "Controls", 179, 179, nothing)
    cv2.createTrackbar("S High", "Controls", 255, 255, nothing)
    cv2.createTrackbar("V High", "Controls", 255, 255, nothing)
    cv2.createTrackbar("Min Area", "Controls", 600, 5000, nothing)


def get_hsv_bounds():
    h_low = cv2.getTrackbarPos("H Low", "Controls")
    s_low = cv2.getTrackbarPos("S Low", "Controls")
    v_low = cv2.getTrackbarPos("V Low", "Controls")
    h_high = cv2.getTrackbarPos("H High", "Controls")
    s_high = cv2.getTrackbarPos("S High", "Controls")
    v_high = cv2.getTrackbarPos("V High", "Controls")
    min_area = cv2.getTrackbarPos("Min Area", "Controls")

    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
    return lower, upper, max(min_area, 50)


def create_kalman_filter(dt=1.0):
    kf = cv2.KalmanFilter(4, 2)

    # State: [x, y, vx, vy]
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], np.float32)

    # Measurement: [x, y]
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], np.float32)

    # Process noise covariance
    kf.processNoiseCov = np.array([
        [1e-2, 0, 0, 0],
        [0, 1e-2, 0, 0],
        [0, 0, 5e-2, 0],
        [0, 0, 0, 5e-2]
    ], np.float32)

    # Measurement noise covariance
    kf.measurementNoiseCov = np.array([
        [1e-1, 0],
        [0, 1e-1]
    ], np.float32)

    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.zeros((4, 1), dtype=np.float32)
    return kf


def detect_object(frame, lower_hsv, upper_hsv, min_area):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None, mask

    ((x, y), radius) = cv2.minEnclosingCircle(largest)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    detection = {
        "center": (cx, cy),
        "radius": int(radius),
        "contour": largest,
        "area": area,
    }
    return detection, mask


def predict_future_points(kf, steps=10):
    state = kf.statePost.copy()
    F = kf.transitionMatrix.copy()
    pts = []

    for _ in range(steps):
        state = F @ state
        pts.append((int(state[0, 0]), int(state[1, 0])))
    return pts


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open the camera. Try closing FaceTime/Zoom/Photo Booth and allow camera access in macOS settings.")
        return

    create_hsv_trackbars()
    kf = create_kalman_filter(dt=1.0)

    measured_path = deque(maxlen=200)
    filtered_path = deque(maxlen=200)
    predicted_path = deque(maxlen=200)

    initialized = False
    lost_frames = 0
    detection_count = 0

    # For prediction evaluation
    next_frame_prediction = None
    last_error = None
    errors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab a frame from the camera.")
            break

        frame = cv2.flip(frame, 1)
        lower_hsv, upper_hsv, min_area = get_hsv_bounds()

        detection, mask = detect_object(frame, lower_hsv, upper_hsv, min_area)

        # Predict with Kalman filter only after initialization
        if initialized:
            pred = kf.predict()
            pred_pt = (int(pred[0, 0]), int(pred[1, 0]))
            predicted_path.append(pred_pt)

        if detection is not None:
            cx, cy = detection["center"]
            radius = detection["radius"]
            contour = detection["contour"]
            current_measured = (cx, cy)

            # Compare previous frame's +1 prediction with current measured position
            if next_frame_prediction is not None and detection_count >= WARMUP_DETECTIONS:
                last_error = euclidean_distance(current_measured, next_frame_prediction)
                errors.append(last_error)

                # Draw previous next-frame prediction and its error line
                cv2.circle(frame, next_frame_prediction, 7, (0, 255, 255), 2)
                cv2.line(frame, next_frame_prediction, current_measured, (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    "Prev +1 prediction",
                    (next_frame_prediction[0] + 10, next_frame_prediction[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    2
                )

            if not initialized:
                kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                initialized = True
                est_pt = (cx, cy)
            else:
                measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                estimated = kf.correct(measurement)
                est_pt = (int(estimated[0, 0]), int(estimated[1, 0]))

            detection_count += 1
            measured_path.append(current_measured)
            filtered_path.append(est_pt)
            lost_frames = 0

            # Draw detection
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, current_measured, radius, (0, 255, 0), 2)
            cv2.circle(frame, current_measured, 4, (0, 255, 0), -1)
            cv2.putText(
                frame,
                "Measured",
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # Draw Kalman estimate
            cv2.circle(frame, est_pt, 5, (255, 0, 0), -1)
            cv2.putText(
                frame,
                "Kalman",
                (est_pt[0] + 10, est_pt[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

            # Predict future points from corrected state
            future_pts = predict_future_points(kf, steps=8)

            # Save 1-step future prediction for evaluation at next frame
            if len(future_pts) > 0:
                next_frame_prediction = future_pts[0]

            for i, pt in enumerate(future_pts):
                cv2.circle(frame, pt, max(1, 5 - i // 2), (0, 0, 255), -1)
                if i > 0:
                    cv2.line(frame, future_pts[i - 1], pt, (0, 0, 255), 2)

        else:
            lost_frames += 1
            next_frame_prediction = None  # cannot evaluate next-frame prediction if object is lost
            cv2.putText(
                frame,
                "Object not detected",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Draw path lines
        for i in range(1, len(measured_path)):
            cv2.line(frame, measured_path[i - 1], measured_path[i], (0, 255, 0), 2)

        for i in range(1, len(filtered_path)):
            cv2.line(frame, filtered_path[i - 1], filtered_path[i], (255, 0, 0), 2)

        for i in range(1, len(predicted_path)):
            cv2.line(frame, predicted_path[i - 1], predicted_path[i], (0, 255, 255), 1)

        # Metrics on screen
        if detection_count < WARMUP_DETECTIONS:
            status_text = f"Warming up... detections: {detection_count}/{WARMUP_DETECTIONS}"
            cv2.putText(frame, status_text, (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        else:
            if last_error is not None:
                cv2.putText(frame, f"Last prediction error: {last_error:.2f} px", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            if len(errors) > 0:
                avg_error = sum(errors) / len(errors)
                cv2.putText(frame, f"Average prediction error: {avg_error:.2f} px", (20, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(frame, f"Evaluated frames: {len(errors)}", (20, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.putText(frame, "Green: measured  Blue: filtered  Red: future  Yellow: prev +1 pred", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.putText(frame, "Press q to quit | c to clear", (20, frame.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("Kalman Filter Object Tracking", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            measured_path.clear()
            filtered_path.clear()
            predicted_path.clear()
            kf = create_kalman_filter(dt=1.0)
            initialized = False
            lost_frames = 0
            detection_count = 0
            next_frame_prediction = None
            last_error = None
            errors.clear()

    cap.release()
    cv2.destroyAllWindows()

    if len(errors) > 0:
        avg_error = sum(errors) / len(errors)
        print(f"\nAverage next-frame prediction error: {avg_error:.2f} pixels")
        print(f"Evaluated frames: {len(errors)}")
    else:
        print("\nNo prediction error was evaluated.")


if __name__ == "__main__":
    main()
