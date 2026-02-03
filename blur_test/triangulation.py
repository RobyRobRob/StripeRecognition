import numpy as np

# Kamera
cam_in  = (700, 700, 320, 240)      # fx, fy, cx, cy
cam_ex  = (-0.6, 0.1, 0.0, 0, 0, 0)  # tx, ty, tz, rx, ry, rz

# Projektor
proj_in  = (850, 850, 400, 300)     # fx, fy, cx, cy
proj_ex  = (0.6, -0.1, 0.0, 0, 8, 0) # tx, ty, tz, rx, ry, rz

# Bildpunkte Test
cam_pxl = [(300, 270), (305, 270)]   # u, v
proj_pxl = [(445, 315), (450, 315)]  # u, v

def local_beam(u, v, intr):
    x = (u - intr[2])/intr[0]
    y = (v - intr[3])/intr[1]
    z = 1
    
    norm = np.sqrt(x**2 + y**2 + z)
    d = [x/norm, y/norm, z/norm]
    
    return d

def world_beam(d, extr):
    rx, ry, rz = np.deg2rad(extr[3:])
    R = Rz(rz) @ Ry(ry) @ Rx(rx)
    d = d @ R
    return d

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c]
    ])

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1]
    ])

def find_point(extr1, extr2, d, e):
    d1 = np.array(d)
    d2 = np.array(e)
    kor1 = np.array(extr1[:3])
    kor2 = np.array(extr2[:3])
    w = kor1 - kor2
    
    a = np.dot(d1,d1)
    b = np.dot(d1,d2)
    c = np.dot(d2,d2)
    d0 = np.dot(d1, w)
    e0 = np.dot(d2, w)
    
    delta = a*c - b**2
    if delta == 0:
        print("kein Punkt, geraden sind parallel")
        return 0
    
    t_near = (b*e0 - c*d0)/delta
    s_near = (a*e0 - b*d0)/delta
    
    X1 = t_near * d1 + kor1
    X2 = s_near * d2 + kor2
    
    return (X1 + X2)/2

def calculate_point(cam, proj):
    d = local_beam(cam[0], cam[1], cam_in)
    d = world_beam(d, cam_ex)
    e = local_beam(proj[0],proj[1], proj_in)
    e = world_beam(e, proj_ex)
    return find_point(cam_ex, proj_ex, d, e)

def calculate_all_points(cam, proj):
    points = []
    for c, p in cam, proj:
        points.append(calculate_point(c, p))
    return points

def save_ply_points(filename, cam_pxl, proj_pxl):
    points = np.array(calculate_all_points(cam_pxl, proj_pxl))
    assert points.ndim == 2 and points.shape[1] == 3

    n = points.shape[0]
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")

save_ply_points("test_points.ply", cam_pxl, proj_pxl)