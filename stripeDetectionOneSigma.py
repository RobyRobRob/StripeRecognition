# Beispiel: Graustufenbild einlesen
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def read_image(img_name):
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, "images", img_name)
    # --- Bild in FARBE und GRAU einlesen ---
    I_color = cv2.imread(img_path)  # Farbbild
    I  = cv2.cvtColor(I_color, cv2.COLOR_BGR2GRAY)  # für Verarbeitung
    return I, I_color

def img_blurring(I):
    # Glätten mit Sigma = 1
    I_sigma = cv2.GaussianBlur(I, (0, 0), sigmaX=1, sigmaY=1)
    return I_sigma

def img_lambda(I_sigma):
    # 2. Ableitungen
    Ixx = cv2.Sobel(I_sigma, cv2.CV_64F, 2, 0, ksize=3)  # 2x in X-Richtung
    Iyy = cv2.Sobel(I_sigma, cv2.CV_64F, 0, 2, ksize=3)  # 2x in Y-Richtung
    Ixy = cv2.Sobel(I_sigma, cv2.CV_64F, 1, 1, ksize=3)  # gemischt

    # Eigenwerte der 2x2-Hessian-Matrix:
    tmp1 = (Ixx + Iyy) / 2.0
    tmp2 = np.sqrt(((Ixx - Iyy) / 2.0)**2 + Ixy**2)

    lambda1 = tmp1 + tmp2
    lambda2 = tmp1 - tmp2
    
    return lambda1, lambda2

def img_vesselness(lambda1, lambda2, alpha, beta):
    # Um Division durch 0 zu vermeiden:
    eps = 1e-8
    # Vesselness berechnen
    V = np.sign(-lambda1) * \
        np.exp(-alpha * np.abs(lambda2 / (lambda1 + eps))) * \
        (1 - np.exp(-beta * (lambda1**2 + lambda2**2)))

    # Optional: auf [0,1] normalisieren, um besser zu visualisieren
    V_norm = (V - np.min(V)) / (np.max(V) - np.min(V))
    return V_norm

def plot_stripes(V_norm):
    plt.figure(figsize=(10,5))
    plt.imshow(V_norm, cmap='gray')
    plt.title("Vesselness Map (Streifen erkannt)")
    plt.axis('off')
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, "images\output_stripes.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')

def detect_stripes(V_norm):
    # ===============================
    # Scanlinie & Peak-Erkennung
    # ===============================

    # 1 Scanlinie wählen (z. B. mittig)
    y_scan = V_norm.shape[0] // 2

    # 2️ Werte entlang der Scanlinie holen
    line_vessel = V_norm[y_scan, :]

    # 3️ Peaks finden (Streifenmitten)
    # "distance" steuert den minimalen Abstand zwischen Peaks
    peaks, props = find_peaks(line_vessel, distance=5, height=0.3)

    print(f"Gefundene Peaks: {len(peaks)}")
    return peaks, props, y_scan

def mark_stripes():
    # ===============================
    # Visualisierung
    # ===============================

    plt.figure(figsize=(12, 6))

    # Original Graustufenbild anzeigen
    # OpenCV -> RGB konvertieren, da OpenCV BGR nutzt
    I_rgb = cv2.cvtColor(I_color, cv2.COLOR_BGR2RGB)

    plt.imshow(I_rgb)
    plt.title("Scanlinie und erkannte Streifen")

    # Scanlinie einzeichnen
    plt.axhline(y_scan, color='lime', linestyle='--', linewidth=1)

    # Peaks als rote Kreuze markieren
    for x in peaks:
        plt.plot(x, y_scan, 'rx', markersize=4, markeredgewidth=1)

    plt.axis('off')
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, "images\output_peaks.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')

def color_detection():
    # Farberkennung

    # Beispielstreifen 
    x = 5
    y = y_scan

    # peaks[x]
    I_rgb = cv2.cvtColor(I_color, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.imshow(I_rgb)

    for i in range(-2, 3):
        plt.plot(peaks[x+i], y_scan, 'rx', markersize=4, markeredgewidth=1)
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, "images\window.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')

    color_min = [300, 300, 300]
    color_max = [0, 0, 0]

    for i in range(-2, 3):
        for j in range(3):
            if(I_rgb[y, peaks[x + i]][j] < color_min[j]):
                color_min[j] = I_rgb[y, peaks[x + i]][j]
            
            if(I_rgb[y, peaks[x + i]][j] > color_max[j]):
                color_max[j] = I_rgb[y, peaks[x + i]][j]
                
    h = [0, 0, 0]
    ambient = [0, 0, 0]
    for i in range(3):
        h[i] = color_max[i] - color_min[i] 
        ambient[i] = color_min[i]

    color = [0, 0, 0]
    for j in range(3):
        color[j] = (I_rgb[y, peaks[x]][j] - ambient[j]) / h[j]
    return ambient, I_rgb, y, x, color, h

def print_values(ambient, I_rgb, y, x, color, h):
    print("==============")
    print("h: " + str(h))
    print("ambient: " + str(ambient))
    print("pixel: " + str(I_rgb[y, peaks[x]]))
    print("==============")
    print(color)
    print("==============")

def color_mapping(top_limit, low_limit):
    if((color[0] > top_limit) & (color[1] < low_limit) & (color[2] < low_limit)):
        print("R")
    elif((color[0] < low_limit) & (color[1] > top_limit) & (color[2] < low_limit)):
        print("G")
    elif((color[0] < low_limit) & (color[1] < low_limit) & (color[2] > top_limit)):
        print("B")
    elif((color[0] > top_limit) & (color[1] > top_limit) & (color[2] < low_limit)):
        print("M")
    elif((color[0] > top_limit) & (color[1] < low_limit) & (color[2] > top_limit)):
        print("Y")
    elif((color[0] < low_limit) & (color[1] > top_limit) & (color[2] > top_limit)):
        print("C")
    else:
        print("?")


I, I_color = read_image("testStripes1.jpg")
I_sigma = img_blurring(I)
lambda1, lambda2 = img_lambda(I_sigma)
V_norm = img_vesselness(lambda1, lambda2, 0.5, 0.5)
plot_stripes(V_norm)
peaks, props, y_scan = detect_stripes(V_norm)
mark_stripes()
ambient, I_rgb, y, x, color, h = color_detection()
print_values(ambient, I_rgb, y, x, color, h)
color_mapping(0.5, 0.2)