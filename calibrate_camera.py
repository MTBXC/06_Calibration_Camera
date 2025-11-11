import numpy as np
import cv2
import glob
import os

# --- CONFIGURATION ---
# Define the size of the chessboard used for calibration
CHESSBOARD_SIZE = (7, 7) # Inner corners count 
SQUARE_SIZE_MM = 17.00    # Real-world size of one square side (e.g., 20.0 mm)
OUTPUT_FILE = 'camera_params.npz'
IMAGE_FOLDER = 'calib_images' # The folder where you saved the images

# Criteria for the SubPixel algorithm (improves corner detection accuracy)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# --- OBJECT POINTS PREPARATION ---
# Prepare "object points" (3D points). Assuming the chessboard is on Z=0 plane.
# objp will be a 3D grid representing the real-world coordinates of the corners.
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
# Generates coordinates (x, y, z) in real-world space, scaled by SQUARE_SIZE_MM
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

# Arrays to store all object points (3D) and image points (2D) from all images
objpoints = []  # 3D point in real world space
imgpoints = []  # 2D points in image plane (pixel coordinates)

# --- IMAGE PROCESSING ---
print("--- Starting Calibration Image Processing ---")

# Load all images from the calib_images folder
images = glob.glob(os.path.join(IMAGE_FOLDER, 'calib_image_*.png'))

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: Could not read image {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the corners on the chessboard pattern
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    # If corners are found, add object points and refine image points
    if ret == True:
        objpoints.append(objp)

        # Refine corner positions to sub-pixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Optional: Display the detected corners (may fail due to GTK error, but try)
        try:
            img = cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
            cv2.imshow('Detected Corners', img)
            cv2.waitKey(100) # Short delay for viewing
        except cv2.error:
            # Continue processing if display fails (due to GTK error)
            pass 
    else:
        print(f"❌ Failed to find chessboard corners in: {fname}")

cv2.destroyAllWindows()

# --- CALIBRATION CALCULATION ---
if len(imgpoints) > 5: # Need a sufficient number of valid images (min ~5-10)
    print(f"\n✅ Found {len(imgpoints)} valid images. Calculating calibration...")
    
    # The core calibration function:
    # It returns: ret, camera_matrix (mtx), distortion_coefficients (dist), 
    # rotation_vectors (rvecs), translation_vectors (tvecs)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    if ret:
        print("\n--- Calibration Results ---")
        # Camera Matrix K: focal lengths (fx, fy) and principal point (cx, cy)
        # $K = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}$
        print("\nCamera Matrix (Intrinsic Matrix - K):")
        print(mtx) 
        
        # Distortion Coefficients D: radial (k1, k2, k3) and tangential (p1, p2)
        # $D = (k_1, k_2, p_1, p_2, k_3)$
        print("\nDistortion Coefficients (D):")
        print(dist)

        # Calculate Reprojection Error (a measure of accuracy)
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        avg_error = mean_error/len(objpoints)
        print(f"\nAverage Reprojection Error: {avg_error:.4f} pixels")
        
        # --- SAVING RESULTS ---
        np.savez(OUTPUT_FILE, mtx=mtx, dist=dist)
        print(f"\n✅ Calibration parameters saved to '{OUTPUT_FILE}'")
        
        # Reprojection Error Guidelines:
        if avg_error > 1.0:
            print("❗ NOTE: Reprojection error is high. Consider re-taking calibration images.")
        elif avg_error < 0.5:
             print("👍 EXCELLENT: Reprojection error is low, calibration is highly accurate.")

    else:
        print("\n❌ ERROR: Calibration calculation failed.")
else:
    print("❌ ERROR: Not enough valid images found for calibration (need at least 5-10).")