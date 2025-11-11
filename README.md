# 📷 IMX219 Camera Calibration Project (NVIDIA Jetson)

This repository contains the necessary scripts and parameters for accurately calibrating the IMX219 camera module used on a NVIDIA Jetson platform.

## Key Features

* **Accurate Calibration:** Achieved an **Average Reprojection Error of 0.0521 pixels**, ensuring professional-grade geometric precision.
* **Parameter Storage:** Calibration matrix (`K`) and distortion coefficients (`D`) are saved in `camera_params.npz`.
* **Real-time Undistortion:** Includes a script to apply live distortion correction to the camera feed using the efficient `cv2.remap` method.

## Project Files

| File | Description |
| :--- | :--- |
| `calibrate_camera.py` | Main script for calculating camera intrinsics using captured chessboard images. |
| `capture_images.py` | Script to capture calibration images (PNG, 1920x1080) from the CSI camera. |
| `live_undistort.py` | Real-time script for applying distortion correction using saved parameters. |
| `camera_params.npz` | **Final Calibration Results** (Matrix K and coefficients D). |

## Setup and Usage

1.  **Calibration:** Run `python calibrate_camera.py` after collecting images in the `calib_images` folder.
2.  **Live Feed:** Run `python live_undistort.py` to see the real-time processing of the corrected video stream.