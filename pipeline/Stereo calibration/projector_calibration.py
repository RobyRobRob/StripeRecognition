import cv2
import numpy as np
import glob
import os

# ================================
# Parameter
# ================================
pattern_size = (9, 6)
square_size = 30.0
projector_resolution = (1024, 768)

# ================================
# 3D Schachbrett Punkte
# ================================
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0],
                       0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# ================================
# Projektor Pattern einmal laden
# ================================
pattern_images = sorted(glob.glob("Greycode pattern/*"))

pattern_stack = []
for path in pattern_images:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    pattern_stack.append(img)

pattern_stack = np.array(pattern_stack)
pattern_stack = (pattern_stack > 127).astype(np.uint8)

num_patterns, H, W = pattern_stack.shape
flat_patterns = pattern_stack.reshape(num_patterns, -1).T

projector_dict = {}
for idx, bits in enumerate(flat_patterns):
    key = tuple(bits)
    u = idx % W
    v = idx // W
    projector_dict.setdefault(key, []).append((u, v))

# ================================
# Listen für Kalibrierung
# ================================
objectPoints = []
imagePoints = []

# ================================
# Alle Pose-Ordner durchlaufen
# ================================
pose_folders = sorted(glob.glob("Greycode images/Pose*"))

for pose_path in pose_folders:

    images = sorted(glob.glob(os.path.join(pose_path, "*")))
    if len(images) == 0:
        continue

    # --- Schachbrett erkennen ---
    img = cv2.imread(images[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not ret:
        continue

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    camera_pixels = corners.reshape(-1,2).astype(np.float32)

    # --- Kamera Bitfolgen ---
    coords = np.round(camera_pixels).astype(int)
    xs = coords[:,0]
    ys = coords[:,1]

    bit_sequences = []
    for img_path in images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        bits = (img[ys, xs] > 127).astype(np.uint8)
        bit_sequences.append(bits)

    bit_sequences = np.stack(bit_sequences, axis=1)

    # --- Matching ---
    projector_points = []
    valid_indices = []

    for i, seq in enumerate(bit_sequences):
        key = tuple(seq)
        if key in projector_dict:
            projector_points.append(projector_dict[key][0])
            valid_indices.append(i)

    if len(projector_points) < 10:
        continue

    projector_points = np.array(projector_points, dtype=np.float32)
    objp_valid = objp[valid_indices]

    objectPoints.append(objp_valid)
    imagePoints.append(projector_points)

# ================================
# Projektor Kalibrierung
# ================================
ret, Kp, dist, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints,
    imagePoints,
    projector_resolution,
    None,
    None
)

print("Reprojection Error:", ret)
print("Kp:\n", Kp)
print("dist:\n", dist)


# ================================
# Stereo Kalibrierung
# ================================

ret, Kc, dist_c, Kp, dist_p, R, T, E, F = cv2.stereoCalibrate(
    objectPoints,
    camera_imagePoints,
    projector_imagePoints,
    Kc,
    dist_c,
    Kp,
    dist_p,
    projector_resolution,
    flags=cv2.CALIB_FIX_INTRINSIC
)

print("Stereo RMS:", ret)
print("Rotation:\n", R)
print("Translation:\n", T)