import stripeDetectionManySigma as sd
import numpy as np
import cv2

def save_ply(filename, points):
    points = np.asarray(points)
    n = points.shape[0]

    with open(filename, "w") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # Punkte
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

    print(f"PLY gespeichert: {filename}")

data = np.load("Stereo calibration\calibration_data.npz")

Kc = data["Kc"]
dist_c = data["dist_c"]
Kp = data["Kp"]
dist_p = data["dist_p"]
R = data["R"]
T = data["T"]
E = data["E"]
F = data["F"]

# Kamera Projektionsmatrix
P1 = Kc @ np.hstack((np.eye(3), np.zeros((3,1))))

# Projektor Projektionsmatrix
P2 = Kp @ np.hstack((R, T))

print("Kalibrierungsdaten geladen.")


cam_pxl, proj_pxl = sd.main()

pts_cam = np.array(cam_pxl, dtype=np.float64).T
pts_proj = np.array(proj_pxl, dtype=np.float64).T

pts_cam = cv2.undistortPoints(
    np.array(cam_pxl, dtype=np.float64).reshape(-1,1,2),
    Kc, dist_c, P=Kc
).reshape(-1,2).T

pts_proj = cv2.undistortPoints(
    np.array(proj_pxl, dtype=np.float64).reshape(-1,1,2),
    Kp, dist_p, P=Kp
).reshape(-1,2).T

points_4d = cv2.triangulatePoints(P1, P2, pts_cam, pts_proj)
points_3d = (points_4d[:3] / points_4d[3]).T

print("Triangulation fertig")

save_ply("pointcloud.ply", points_3d)