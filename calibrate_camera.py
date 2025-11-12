import numpy as np
import cv2
import glob
import os

# --- CONFIGURATION ---
CHESSBOARD_SIZE = (7, 7) # Inner corners count 
SQUARE_SIZE_MM = 17.00   # Real-world size of one square side (mm)
OUTPUT_FILE = 'camera_params.npz'
IMAGE_FOLDER = 'calib_images'

# Criteria for SubPixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# --- OBJECT POINTS PREPARATION ---
# Prepare 3D points (x, y, z=0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

# Storage for 3D points (objpoints) and 2D points (imgpoints)
objpoints = []
imgpoints = []

# --- IMAGE PROCESSING ---
print("--- Starting Calibration Image Processing ---")

images = glob.glob(os.path.join(IMAGE_FOLDER, 'calib_image_*.png'))

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: Could not read image {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret == True:
        objpoints.append(objp)

        # Sub-pixel refinement
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Display detected corners
        try:
            img = cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
            cv2.imshow('Detected Corners', img)
            cv2.waitKey(100)
        except cv2.error:
            pass # Ignore display errors
    else:
        print(f"Failed to find chessboard corners in: {fname}")

cv2.destroyAllWindows()

# --- CALIBRATION CALCULATION ---
if len(imgpoints) > 5:
    print(f"\nFound {len(imgpoints)} valid images. Calculating calibration...")
    
    # Core calibration function
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    if ret:
        print("\n--- Calibration Results ---")
        print("\nCamera Matrix (K):")
        print(mtx) 
        
        print("\nDistortion Coefficients (D):")
        print(dist)

        # Calculate Reprojection Error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        avg_error = mean_error/len(objpoints)
        print(f"\nAverage Reprojection Error: {avg_error:.4f} pixels")
        
        # --- SAVING RESULTS ---
        np.savez(OUTPUT_FILE, mtx=mtx, dist=dist)
        print(f"\nCalibration parameters saved to '{OUTPUT_FILE}'")
        
        # Simple error assessment
        if avg_error > 1.0:
            print("NOTE: Reprojection error is high.")
        elif avg_error < 0.5:
             print("EXCELLENT: Calibration highly accurate.")

    else:
        print("\nERROR: Calibration calculation failed.")
else:
    print("ERROR: Not enough valid images found.")