# Beispiel: Graustufenbild einlesen
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import os
import time

test_img_num = 6
sigmas = [1, 2, 3]
save_plot = True
save_img = False
one_line = True

def plot_array(arr, name, multiple):
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, "plots/" + name)
    plt.figure(figsize=(10,5))
    plt.title(name)
    plt.xlabel("x-Pixel")
    plt.ylabel("Helligkeit (0–255)")
    plt.grid(True)
    if multiple:
        for i in range(len(arr)):
            y_scan = arr[i].shape[0] // 2
            plt.plot(arr[i][y_scan, :], label="Kurve " + str(i+1))
        plt.legend(loc="lower left")
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        return arr
    else:
        y_scan = arr.shape[0] // 2
        Img_line = arr[y_scan, :]
        plt.plot(Img_line)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        return Img_line

def plot_array_over_img(arr, name, multiple, img):
    folder = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(folder, "plots/" + name)
    
    plt.title(name)
    plt.xlabel("x-Pixel")
    plt.ylabel("Helligkeit (0–255)")
    plt.grid(True)
    if multiple:
        y_scan = arr[0].shape[0] // 2
        I_RGB_line = img[y_scan]
        max_val = max(arr[0][y_scan, :])
        if max_val < 1:
            max_val = 1
        plt.figure(figsize=(10,10))
        I_RGB_stripe = np.stack([I_RGB_line] * int(max_val), axis=0)
        
        plt.imshow(I_RGB_stripe, cmap=None)
        for i in range(len(arr)):
            plt.plot(arr[i][y_scan, :], label="Kurve " + str(i+1))
        plt.legend(loc="center left")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return arr
    else:
        if arr.ndim > 1:
            y_scan = arr.shape[0] // 2
        else:
            y_scan = img.shape[0]//2 
        I_RGB_line = img[y_scan]
        max_val = max(arr[y_scan, :])
        if max_val < 1:
            max_val = 1
        plt.figure(figsize=(10,10))
        I_RGB_stripe = np.stack([I_RGB_line] * int(max_val), axis=0)
    
        plt.imshow(I_RGB_stripe, cmap=None)
        if arr.ndim > 1:
            Img_line = arr[y_scan, :]
        else:
            Img_line = arr
        plt.plot(Img_line)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return Img_line

def read_image(img_name):       # Img einlesen
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, "images", "testStripes" + str(img_name) + ".jpg")
    I_BGR = cv2.imread(img_path)
    I_GRAY  = cv2.cvtColor(I_BGR, cv2.COLOR_BGR2GRAY)
    return I_GRAY, I_BGR

def img_blurring(I_GRAY, sigmas):  # Img blurren
    I_sigmas = []
    # Glätten mit Sigma
    count = 1
    for i in sigmas:
        ksize = int(6 * i + 1)
        I_sigma = cv2.GaussianBlur(I_GRAY, (ksize, ksize), sigmaX=i, sigmaY=i)             # (5, 5) Kernel, gibt Feldgröße an, sigma, gibt gewichtung an
        # I_sigma = cv2.blur(I, (3, 3))
        I_sigmas.append(I_sigma)
        if save_img:
            plt.figure(figsize=(12, 6))
            plt.imshow(I_sigma, cmap='gray')  # <<< WICHTIG
            plt.title(f"img{test_img_num}_blurred   Sigma: {sigmas[count-1]}")
            plt.axis('off')

            folder = os.path.dirname(os.path.abspath(__file__))
            img_path = os.path.join(folder, f"images/img{test_img_num}_blurred_{sigmas[count-1]}.png")
            count+=1

            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
    if save_plot:
        plot_array_over_img(I_sigmas, "img" + str(test_img_num) + "_Sigmas", True, I_RGB)
    return I_sigmas

def img_lambda(I_sigmas):       # Lambdas berechnen
    lambdas = []
    count = 0
    Ixxs = []
    Iyys = []
    Ixys = []
    for I_sigma in I_sigmas:
        Iyy = gaussian_filter(I_sigma, sigma=sigmas[count], order=(2,0))
        Ixx = gaussian_filter(I_sigma, sigma=sigmas[count], order=(0,2))
        Ixy = gaussian_filter(I_sigma, sigma=sigmas[count], order=(1,1))
        
        # da Große Sigma werden systematisch benachteiligt werden
        Ixx *= sigmas[count]**2
        Iyy *= sigmas[count]**2
        Ixy *= sigmas[count]**2
        count += 1
        
        tmp1 = (Ixx + Iyy) / 2.0
        tmp2 = np.sqrt(((Ixx - Iyy)/2)**2 + (Ixy**2))
        lambda1 = tmp1 + tmp2
        lambda2 = tmp1 - tmp2
        lambdas.append((lambda1, lambda2))
        
        # Um alle Ableitungen plotten zu können
        Ixxs.append(Ixx)
        Iyys.append(Iyy)
        Ixys.append(Ixy)
    plot_array_over_img(Ixxs, "2_Ableitungen", True, I_RGB)
    return lambdas

def img_vesselness(lambdas, alpha, beta): # erstellt vesselness map
    eps = 1e-8
    V_all = []

    for lambda1, lambda2 in lambdas:
        # 1) Sortieren: |λ1| >= |λ2|
        l1 = lambda1
        l2 = lambda2
        switch = np.abs(lambda2) > np.abs(lambda1)
        l1, l2 = np.where(switch, lambda2, lambda1), np.where(switch, lambda1, lambda2)
        
        # 2) Vesselness-Formel
        mask = l1 > 0
        R = abs(l2 / (l1 + eps))
        S = l1**2 + l2**2
        V = np.zeros_like(l1)
        V[mask] = np.exp(-alpha * R[mask]) * (1 - np.exp(-beta * S[mask]))
        
        V_all.append(V)
    #plot_array(V_all,"img" + str(test_img_num) + "_V_all.png", True)
    plot_array_over_img(V_all, "img" + str(test_img_num) + "_V_all_img", True, I_RGB)
    
    # Multi-scale Vesselness -> Pixelweises Maximum über alle sigma
    V_all = np.stack(V_all, axis=0)
    V_max = np.max(V_all, axis=0)
    
    if save_plot:
        #plot_array(V_max, "img" + str(test_img_num) + "_V_max.png", False)
        plot_array_over_img(V_max, "img" + str(test_img_num) + "_V_max_img", False, I_RGB)

    # Normierung
    V_norm = (V_max - np.min(V_max)) / (np.max(V_max) - np.min(V_max) + eps)
    #if save_plot:
        #plot_array(V_norm, "img" + str(test_img_num) + "_V_norm.png", False)

    return V_norm

def plot_stripes(V_norm):   # stellt vessellnesmap als schwarz weiß bild dar
    plt.figure(figsize=(10,5))
    plt.title("Vesselness Map (Streifen erkannt)")
    plt.axis('off')
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, "images\output_stripes.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')

def detect_stripes(V_norm): # scannt Veselness map nach streifen ab und gibt koordinaten zurück
    # 1 Scanlinie wählen (z. B. mittig)
    y_scan = V_norm.shape[0] // 2

    # 2️ Werte entlang der Scanlinie holen
    line_vessel = V_norm[y_scan, :]

    # 3️ Peaks finden (Streifenmitten)
    # "distance" steuert den minimalen Abstand zwischen Peaks und height die höhe die ein pixel haben muss um ein peak zu sein
    peaks, props = find_peaks(line_vessel, distance=10, height=0.2)

    print(f"Gefundene Peaks: {len(peaks)}")
    I_rgb = cv2.cvtColor(I_BGR, cv2.COLOR_BGR2RGB)
    return peaks, props, y_scan, I_rgb

def mark_stripes():     # markiert auf dem Bild die erkannten peaks
    # ===============================
    # Visualisierung
    # ===============================

    plt.figure(figsize=(12, 6))

    # Original Graustufenbild anzeigen
    # OpenCV -> RGB konvertieren, da OpenCV BGR nutzt
    I_RGB = cv2.cvtColor(I_BGR, cv2.COLOR_BGR2RGB)
    
    I_RGB_line = I_RGB
    for i in range(len(I_RGB)):
        I_RGB_line[i] = I_RGB[y_scan]
    
    plt.imshow(I_RGB_line)
    plt.title("Irgb y_scan line")
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, "plots\\1_I_rgb_line.png")
    plt.plot(plot_array(I_RGB, "img" + str(test_img_num) + "_Original.png", False))
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.imshow(I_RGB)
    plt.title("Scanlinie und erkannte Streifen")

    # Scanlinie einzeichnen
    plt.axhline(y_scan, color='lime', linestyle='--', linewidth=1)

    # Peaks als rote Kreuze markieren
    for x in peaks:
        plt.plot(x, y_scan, 'rx', markersize=3, markeredgewidth=1)

    plt.axis('off')
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, "images\output_peaks.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')

def mark_window(x, y_scan): # markiert die sreifen in einem Bild
    # peaks[x]
    I_RGB = cv2.cvtColor(I_BGR, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.imshow(I_RGB)

    for i in range(-2, 3):
        plt.plot(peaks[x+i], y_scan, 'rx', markersize=4, markeredgewidth=1)
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, "images\window.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

def color_detection(I_RGB, y_scan, x, peaks):  # erkennt die Farbwerte an der position anhand des windows
    color_min = [300, 300, 300]
    color_max = [0, 0, 0]
    buffer = 0
    if x > (len(peaks) - 3):
        buffer = 3 - (len(peaks) - x)
    if x < 2:
        buffer = -2 + x

    for i in range(-2 - buffer, 3 - buffer):
        for j in range(3):
            if(I_RGB[y_scan, peaks[x + i]][j] < color_min[j]):
                color_min[j] = I_RGB[y_scan, peaks[x + i]][j]
            
            if(I_RGB[y_scan, peaks[x + i]][j] > color_max[j]):
                color_max[j] = I_RGB[y_scan, peaks[x + i]][j]
                
    h = [0, 0, 0]
    ambient = [0, 0, 0]
    for i in range(3):
        h[i] = color_max[i] - color_min[i] 
        ambient[i] = color_min[i]

    color = [0, 0, 0]
    for j in range(3):
        color[j] = (I_RGB[y_scan, peaks[x]][j] - ambient[j]) / h[j]
    return ambient, I_RGB, y_scan, x, color, h

def print_values(ambient, I_RGB, y, x, color, h):   # printed erkannte Werte im Window 
    print("==============")
    print("h: " + str(h))
    print("ambient: " + str(ambient))
    print("pixel: " + str(I_RGB[y, peaks[x]]))
    print("==============")
    print(color)
    print("==============")

def color_mapping(color, top_limit, low_limit):     # weißt den farbwerten einer der 6 farben zu
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

def colorline_detection(I_RGB, y):      # erkennt alle farben in einer reihe
    colorcombination = [""] * len(peaks)
    for i in range(len(peaks)):
        # mark_window(x, y_scan)
        ambient, I_RGB, y, x, color, h = color_detection(I_RGB, y, i, peaks)
        # print_values(ambient, I_rgb, y, x, color, h)
        colorcombination[i] = color_mapping(color, 0.5, 0.2)
    return colorcombination

def test_accuracy(colorcombination, test_img):  # testet wie gut die farben erkannt wurden
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



I_GRAY, I_BGR = read_image(test_img_num)
I_RGB = cv2.cvtColor(I_BGR, cv2.COLOR_BGR2RGB)
I_sigmas = img_blurring(I_GRAY, sigmas)

lambdas = img_lambda(I_sigmas)
V_norm = img_vesselness(lambdas, 0.5, 0.5 )

if save_img:
    plot_stripes(V_norm)

peaks, props, y_scan, I_rgb = detect_stripes(V_norm)

if save_img:
    mark_stripes()