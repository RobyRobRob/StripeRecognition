# Beispiel: Graustufenbild einlesen
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import time

test_img_num = 5
sigmas = [3.0, 3.5, 4.0]
save_plot = True  

def read_image(img_name):
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, "images", "testStripes" + str(img_name) + ".jpg")
    I_color = cv2.imread(img_path)
    I  = cv2.cvtColor(I_color, cv2.COLOR_BGR2GRAY)
    return I, I_color

def img_blurring(I, sigmas):
    I_sigmas = []
    # Glätten mit Sigma
    for i in sigmas:
        I_sigma = cv2.GaussianBlur(I, (0, 0), sigmaX=i, sigmaY=i)
        I_sigmas.append(I_sigma)
    return I_sigmas

def img_lambda(I_sigmas):
    lambdas = []
    for I_sigma in I_sigmas:
        Ixx = cv2.Sobel(I_sigma, cv2.CV_64F, 2, 0, ksize=3)  # 2x in X-Richtung
        Iyy = cv2.Sobel(I_sigma, cv2.CV_64F, 0, 2, ksize=3)  # 2x in Y-Richtung
        Ixy = cv2.Sobel(I_sigma, cv2.CV_64F, 1, 1, ksize=3)  # gemischt

        # Eigenwerte der 2x2-Hessian-Matrix:
        tmp1 = (Ixx + Iyy) / 2.0
        tmp2 = np.sqrt(((Ixx - Iyy) / 2.0)**2 + Ixy**2)
        lambda1 = tmp1 + tmp2
        lambda2 = tmp1 - tmp2
        lambdas.append((lambda1, lambda2))
    
    return lambdas

def img_vesselness(lambdas, alpha, beta):
    eps = 1e-8
    V_all = []

    for lambda1, lambda2 in lambdas:
        # Formel aus dem Paper:
        # V = sign(-lambda1) * exp(-alpha*(lambda2/lambda1)^2) * (1 - exp(-beta*(lambda1^2 + lambda2^2)))
        
        R = (lambda2 / (lambda1 + eps)) ** 2
        S = lambda1**2 + lambda2**2

        V = np.sign(-lambda1) * np.exp(-alpha * R) * (1 - np.exp(-beta * S))

        V_all.append(V)

    # Multi-scale Vesselness -> Pixelweises Maximum über alle sigma
    V_all = np.stack(V_all, axis=0)
    V_max = np.max(V_all, axis=0)

    # Normierung
    V_norm = (V_max - np.min(V_max)) / (np.max(V_max) - np.min(V_max) + eps)

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
    # 1 Scanlinie wählen (z. B. mittig)
    y_scan = V_norm.shape[0] // 2

    # 2️ Werte entlang der Scanlinie holen
    line_vessel = V_norm[y_scan, :]

    # 3️ Peaks finden (Streifenmitten)
    # "distance" steuert den minimalen Abstand zwischen Peaks
    peaks, props = find_peaks(line_vessel, distance=5, height=0.3)

    print(f"Gefundene Peaks: {len(peaks)}")
    I_rgb = cv2.cvtColor(I_color, cv2.COLOR_BGR2RGB)
    return peaks, props, y_scan, I_rgb

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

def mark_window(x, y_scan):
    # peaks[x]
    I_rgb = cv2.cvtColor(I_color, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.imshow(I_rgb)

    for i in range(-2, 3):
        plt.plot(peaks[x+i], y_scan, 'rx', markersize=4, markeredgewidth=1)
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, "images\window.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

def color_detection(I_rgb, y_scan, x, peaks):
    color_min = [300, 300, 300]
    color_max = [0, 0, 0]
    buffer = 0
    if x > (len(peaks) - 3):
        buffer = 3 - (len(peaks) - x)
    if x < 2:
        buffer = -2 + x

    for i in range(-2 - buffer, 3 - buffer):
        for j in range(3):
            if(I_rgb[y_scan, peaks[x + i]][j] < color_min[j]):
                color_min[j] = I_rgb[y_scan, peaks[x + i]][j]
            
            if(I_rgb[y_scan, peaks[x + i]][j] > color_max[j]):
                color_max[j] = I_rgb[y_scan, peaks[x + i]][j]
                
    h = [0, 0, 0]
    ambient = [0, 0, 0]
    for i in range(3):
        h[i] = color_max[i] - color_min[i] 
        ambient[i] = color_min[i]

    color = [0, 0, 0]
    for j in range(3):
        color[j] = (I_rgb[y_scan, peaks[x]][j] - ambient[j]) / h[j]
    return ambient, I_rgb, y_scan, x, color, h

def print_values(ambient, I_rgb, y, x, color, h):
    print("==============")
    print("h: " + str(h))
    print("ambient: " + str(ambient))
    print("pixel: " + str(I_rgb[y, peaks[x]]))
    print("==============")
    print(color)
    print("==============")

def color_mapping(color, top_limit, low_limit):
    if((color[0] > top_limit) & (color[1] < low_limit) & (color[2] < low_limit)):
        return "R"
    elif((color[0] < low_limit) & (color[1] > top_limit) & (color[2] < low_limit)):
        return "G"
    elif((color[0] < low_limit) & (color[1] < low_limit) & (color[2] > top_limit)):
        return "B"
    elif((color[0] > top_limit) & (color[1] > top_limit) & (color[2] < low_limit)):
        return "Y"
    elif((color[0] > top_limit) & (color[1] < low_limit) & (color[2] > top_limit)):
        return "M"
    elif((color[0] < low_limit) & (color[1] > top_limit) & (color[2] > top_limit)):
        return "C"
    else:
        return "?"

def colorline_detection(I_rgb):
    colorcombination = [""] * len(peaks)
    for i in range(len(peaks)):
        # mark_window(x, y_scan)
        ambient, I_rgb, y, x, color, h = color_detection(I_rgb, y_scan, i, peaks)
        # print_values(ambient, I_rgb, y, x, color, h)
        colorcombination[i] = color_mapping(color, 0.5, 0.2)
    return colorcombination

def test_accuracy(colorcombination, test_img):
    folder = os.path.dirname(os.path.abspath(__file__))
    Solution_path = os.path.join(folder, "Solutions.txt")
    with open(Solution_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    #print("Solution: ")
    Solution = lines[2*test_img - 1].strip()
    #print(Solution)
    #print("".join(colorcombination))
    errors = 0
    for i in range(len(Solution)):
        if colorcombination[i] != Solution[i]:
            #print(str(i) + ": " + colorcombination[i] + ", " + Solution[i]) 
            errors += 1
    return 100 - (errors/len(colorcombination) * 100)

start = time.time()
I, I_color = read_image(test_img_num)
I_sigmas = img_blurring(I, sigmas)
lambdas = img_lambda(I_sigmas)
V_norm = img_vesselness(lambdas, 0.5, 0.5)
if save_plot:
    plot_stripes(V_norm)
start2 = time.time()
peaks, props, y_scan, I_rgb = detect_stripes(V_norm)
if save_plot:
    mark_stripes()
colorcombination = colorline_detection(I_rgb)

stop = time.time()

print("Accuracy = " + str(test_accuracy(colorcombination, test_img_num)) + " %")
print("Total Time     : " + str(stop - start))
print("Time per stripe: " + str((stop - start2)/1))