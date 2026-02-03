import stripeDetectionManySigma as sd 
import triangulation as tr 




cam_pxl = [(300, 270), (305, 270)]
proj_pxl = [(445, 315), (450, 315)]
tr.save_ply_points("blur_test/test_points.ply", cam_pxl, proj_pxl)