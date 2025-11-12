# Beispiel: Graustufenbild einlesen
import cv2
import numpy as np
import matplotlib.pyplot as plt

I = cv2.imread(r"C:\Users\User\Desktop\Bachelorprojekt\Stripes\testStripes2.jpg", cv2.IMREAD_GRAYSCALE)

# Glätten mit Sigma = 1
I_sigma = cv2.GaussianBlur(I, (0, 0), sigmaX=1, sigmaY=1)

# 2. Ableitungen
Ixx = cv2.Sobel(I_sigma, cv2.CV_64F, 2, 0, ksize=3)  # 2x in X-Richtung
Iyy = cv2.Sobel(I_sigma, cv2.CV_64F, 0, 2, ksize=3)  # 2x in Y-Richtung
Ixy = cv2.Sobel(I_sigma, cv2.CV_64F, 1, 1, ksize=3)  # gemischt

# Eigenwerte der 2x2-Hessian-Matrix:
tmp1 = (Ixx + Iyy) / 2.0
tmp2 = np.sqrt(((Ixx - Iyy) / 2.0)**2 + Ixy**2)

lambda1 = tmp1 + tmp2
lambda2 = tmp1 - tmp2

# Parameter (kannst du anpassen)
alpha = 0.5
beta = 0.5

# Um Division durch 0 zu vermeiden:
eps = 1e-8

# Vesselness berechnen
V = np.sign(-lambda1) * \
    np.exp(-alpha * np.abs(lambda2 / (lambda1 + eps))) * \
    (1 - np.exp(-beta * (lambda1**2 + lambda2**2)))

# Optional: auf [0,1] normalisieren, um besser zu visualisieren
V_norm = (V - np.min(V)) / (np.max(V) - np.min(V))


plt.figure(figsize=(10,5))
plt.imshow(V_norm, cmap='gray')
plt.title("Vesselness Map (Streifen erkannt)")
plt.axis('off')
plt.show()