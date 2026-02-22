import stripeDetectionManySigma as sd 
import triangulation as tr 

cam_pxl, proj_pxl = sd.main()
tr.save_ply_points("blur_test/test_points.ply", cam_pxl, proj_pxl)