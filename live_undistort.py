import numpy as np
import cv2
import os
import time

# --- CONFIGURATION ---
PARAMS_FILE = 'camera_params.npz'
CAMERA_INDEX = 0  # 0 for the default CSI camera

# Use the same 1920x1080 resolution used during calibration
GSTREAMER_PIPELINE = (
    "nvarguscamerasrc sensor-id=%d ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink max-buffers=1 drop=true"
) % CAMERA_INDEX
CAPTURE_SOURCE = GSTREAMER_PIPELINE


# --- Main function ---
def live_undistortion(source=CAPTURE_SOURCE, params_file=PARAMS_FILE):
    """
    Loads calibration parameters and applies real-time distortion correction 
    to the camera feed using the efficient cv2.remap method, with live display.
    """
    
    # 1. Load calibration parameters (Camera Matrix K and Distortion Coefficients D)
    try:
        with np.load(params_file) as X:
            mtx, dist = [X[i] for i in ('mtx', 'dist')]
    except FileNotFoundError:
        print(f"Error: Calibration file '{params_file}' not found. Run calibrate_camera.py first.")
        return
        
    # 2. Camera initialization
    cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Define frame size (must match GStreamer pipeline and calibration resolution)
    w = 1920
    h = 1080

    # 3. Pre-calculate the optimal camera matrix and undistortion maps
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), cv2.CV_32FC1)

    print("\n--- Starting Live Undistortion ---")
    print("-> Press 'q' on the image window to QUIT.")
    print("WARNING: Live display may fail due to GTK initialization errors on remote desktop.")

    # Scale factor for displaying large 1920x1080 frames on screen
    SCALE_FACTOR = 0.5 
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            time.sleep(0.1)
            continue
        
        # 4. Apply real-time correction using pre-calculated maps
        corrected_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        
        # 5. Optional: Resize for display purposes (Original and Corrected)
        display_orig = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        display_corr = cv2.resize(corrected_frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        
        # 6. Display frames
        try:
            cv2.imshow('01 - ORIGINAL (Distorted)', display_orig)
            cv2.imshow('02 - CORRECTED (Undistorted)', display_corr)
        except cv2.error:
             print("GTK Error detected. Display failed. Continuing in headless mode...")
             cv2.destroyAllWindows() 

        # 7. Check for exit key (q)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Processing Finished ---")


if __name__ == "__main__":
    live_undistortion()