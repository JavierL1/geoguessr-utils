import cv2

print("OpenCL Enabled:", cv2.ocl.useOpenCL())
cv2.ocl.setUseOpenCL(True)
print("OpenCL Device Connected:", cv2.ocl.useOpenCL())
