import mrcal 

model = mrcal.cameramodel('./camera-0.cameramodel')
lensmodel, intrinsics_vec = model.intrinsics() 

# intrinsics[0] is the lens model string 
# intrinsics[1] contains [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6] 
fx, fy, cx, cy = intrinsics_vec[:4] 
dist_coeff = intrinsics_vec[4:]

print(f"fx: {fx:.4f}")
print(f"fy: {fy:.4f}")
print(f"cx: {cx:.4f}")
print(f"cy: {cy:.4f}") 
print("Distortion coefficients (D):", dist_coeff) 
