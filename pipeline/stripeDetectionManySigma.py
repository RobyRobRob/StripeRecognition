# Beispiel: Graustufenbild einlesen
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_closing
import os
import re
import time

test_img_num = 5
sigmas = [1, 2, 3]
save_plot = False
save_img = False
one_line = False
#================================================================================================
# Visual functions
#================================================================================================
def save_image(plot, color, title, path):
    plt.figure(figsize=(12, 6))
    plt.imshow(plot, cmap=color)  # <<< WICHTIG
    plt.title(title)
    plt.axis('off')
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, path)
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

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

def plot_array_over_img(arr, name, multiple, img, scale = 1):
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
        I_RGB_stripe = np.stack([I_RGB_line] * int(max_val) * scale, axis=0)
        
        plt.imshow(I_RGB_stripe, cmap=None)
        for i in range(len(arr)):
            plt.plot(arr[i][y_scan, :]*scale, label="Kurve " + str(i+1))
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
        I_RGB_stripe = np.stack([I_RGB_line] * int(max_val)* scale, axis=0)
    
        plt.imshow(I_RGB_stripe, cmap=None)
        if arr.ndim > 1:
            Img_line = arr[y_scan, :]
        else:
            Img_line = arr
        plt.plot(Img_line*scale)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return Img_line

def img_vesselnessmap(V_norm):   # stellt vessellnesmap als schwarz weiß bild dar
    plt.figure(figsize=(10,5))
    plt.title("Vesselness Map (Streifen erkannt)")
    plt.axis('off')
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, "images\output_stripes.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')

def mark_stripes():     # markiert auf dem Bild die erkannten peaks
    plt.figure(figsize=(12, 6))
    I_RGB_line = I_RGB
    for i in range(len(I_RGB)):
        I_RGB_line[i] = I_RGB[y_scan]
    
    # Im plot
    if save_plot:
        plt.imshow(I_RGB_line)
        plt.title("Irgb peaks in y_scan line")
        folder = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(folder, "plots\\peaks_in_graph.png")
        plt.plot(np.max(I_RGB_line[y_scan], axis=1))
        for x in range(len(peaks)):
            plt.plot(peaks[x], np.max(I_RGB[y_scan, peaks[x], :]), 'rx', markersize=3, markeredgewidth=1)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Im Bild
    if save_img:
        plt.figure(figsize=(12, 6))
        plt.imshow(I_RGB_line)
        for i in peaks:
            plt.plot(i, y_scan, 'rx', markersize=4, markeredgewidth=1)
        folder = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(folder, "images\stripes_peaks.png")
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()

def mark_window(x, y_scan): # markiert die sreifen in einem Bild
    plt.figure(figsize=(12, 6))
    plt.imshow(I_RGB)

    for i in range(-2, 3):
        plt.plot(peaks[x+i], y_scan, 'rx', markersize=4, markeredgewidth=1)
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, "images\window.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_values(ambient, I_RGB, y, x, color, h):   # printed erkannte Werte im Window 
    print("==============")
    print("h: " + str(h))
    print("ambient: " + str(ambient))
    print("pixel: " + str(I_RGB[y, peaks[x]]))
    print("==============")
    print(color)
    print("==============")

def compare_img_proj(cam_pxl, proj_pxl):
    folder = os.path.dirname(os.path.abspath(__file__))
    image_path1 = folder + "/images/testStripes1.jpg"
    output_path1 = folder + "/solutions/ProjectorMarks.png"
    
    image1 = cv2.imread(image_path1)

    if image1 is None:
        raise ValueError("Bild konnte nicht geladen werden.")
    
    # === X an jede Koordinate zeichnen ===
    for (x, y) in proj_pxl:
        cv2.putText(image1, "X", (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255,255,255), 3)
    # === Bild speichern ===
    cv2.imwrite(output_path1, image1)
    
    
    image_path2 = folder + "/images/testStripes" + str(test_img_num) + ".jpg"
    output_path2 = folder + "/solutions/ImageMarks.png"
    
    image2 = cv2.imread(image_path2)

    if image2 is None:
        raise ValueError("Bild konnte nicht geladen werden.")

    # === X an jede Koordinate zeichnen ===
    for (x, y) in cam_pxl:
        cv2.putText(image2, "X", (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255,255,255), 3)
    # === Bild speichern ===
    cv2.imwrite(output_path2, image2)
    
    # Breite anpassen, damit beide Bilder die gleiche Breite haben
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    if w1 != w2:
        new_height = int(h2 * w1 / w2)
        image2 = cv2.resize(image2, (w1, new_height))

    # Bilder vertikal stapeln
    combined = np.vstack((image1, image2))

    cv2.imwrite(folder + "/solutions/combined.jpg", combined)

#================================================================================================
# Essential functions
#================================================================================================
def read_image(img_name):       # Img einlesen
    folder = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(folder, "images", "testStripes" + str(img_name) + ".jpg")
    I_BGR = cv2.imread(img_path)
    I_GRAY  = cv2.cvtColor(I_BGR, cv2.COLOR_BGR2GRAY)
    return I_GRAY, I_BGR

def img_blurring(I_GRAY, sigmas):  # Img blurren
    I_sigmas = []
    count = 1
    for i in sigmas:
        ksize = int(6 * i + 1)
        I_sigma = cv2.GaussianBlur(I_GRAY, (ksize, ksize), sigmaX=i, sigmaY=i)  # (5, 5) Kernel, gibt Feldgröße an, sigma, gibt gewichtung an
        I_sigmas.append(I_sigma)
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
    if save_plot and one_line is False:
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
    
    # Multi-scale Vesselness -> Pixelweises Maximum über alle sigma
    V_all = np.stack(V_all, axis=0)
    V_max = np.max(V_all, axis=0)

    # Normierung
    V_norm = (V_max - np.min(V_max)) / (np.max(V_max) - np.min(V_max) + eps)

    return V_norm, V_all, V_max

def get_vessels():
    I_GRAY, I_BGR = read_image(test_img_num)
    I_RGB = cv2.cvtColor(I_BGR, cv2.COLOR_BGR2RGB)
    I_sigmas = img_blurring(I_GRAY, sigmas)
    lambdas = img_lambda(I_sigmas)
    V_norm, V_all, V_max = img_vesselness(lambdas, 0.5, 0.5 )
    return I_RGB, I_BGR, V_norm
#==================================================================================================
def create_projector_values(all_peaks, all_colorcombination):
    if test_img_num == 1:
        with open("blur_test/peaks_projector.txt", "w") as f:
            for x in range(len(all_peaks)):
                f.write(f"{all_peaks[x]}")
                f.write(f"\n")
        with open("blur_test/color_projector.txt", "w") as f:
            for x in range(len(all_colorcombination)):
                f.write(f"{all_colorcombination[x]}")
                f.write(f"\n")

def get_projector_values(window_size):
    if test_img_num != 1:
        file = os.path.dirname(os.path.abspath(__file__))
        projector_points = []
        projector_color = []
        with open(file + "/peaks_projector.txt", "r", encoding="utf-8") as f:
            for line in f:
                projector_points.append(int(line))
        with open(file + "/color_projector.txt", "r", encoding="utf-8") as f:
            for line in f:
                projector_color.append(line.rstrip('\n'))
        
        projector_values = {}
        for i in range(len(projector_color) - window_size + 1):
            key = tuple(projector_color[i: i + window_size])
            value = projector_points[i: i + window_size]
            projector_values[key] = value
        return projector_values
    else:
        return None, None

def find_mask(V_norm, y):
    #print(V_norm[y])
    mask = V_norm[y,:] >= 0.9
    mask_int = mask.astype(np.int8)
    #print(mask)
    diff = np.diff(mask_int)

    intervall_anfang = np.where(diff == 1)[0] + 1
    intervall_ende = np.where(diff == -1)[0] + 1

    if mask[0]:
        intervall_anfang = np.insert(intervall_anfang, 0, 0)
    if mask[-1]:
        intervall_ende = np.append(intervall_ende, len(mask))
    return intervall_anfang, intervall_ende

def detect_stripes(I_RGB_max, y): # scannt nach peaks und filtert mit der vesellnes map
    peaks, props = find_peaks(I_RGB_max[y], prominence= 10, distance=10)
    return peaks

def find_true_stripes(peaks, intervall_anfang, intervall_ende):
    inside = ((peaks[:, None] >= intervall_anfang[None, :]) &(peaks[:, None] <= intervall_ende[None, :]))
    true_peaks = peaks[np.any(inside, axis=1)]
    
    if one_line:
        print(f"Gefundene Peaks: {len(peaks)}")
    return true_peaks

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
    eps = 1e-6
    for j in range(3):
        color[j] = (I_RGB[y_scan, peaks[x]][j] - ambient[j]) / (h[j] + eps)
    return ambient, I_RGB, y_scan, x, color, h

def color_mapping(color, top_limit, low_limit):     # weißt den farbwerten einer der 6 farben zu
    if((color[0] > top_limit) & (color[1] < low_limit) & (color[2] < low_limit)):
        return "R" #1 #R
    elif((color[0] < low_limit) & (color[1] > top_limit) & (color[2] < low_limit)):
        return "G"#2 #G
    elif((color[0] < low_limit) & (color[1] < low_limit) & (color[2] > top_limit)):
        return "B"#3 #B
    elif((color[0] > top_limit) & (color[1] > top_limit) & (color[2] < low_limit)):
        return "Y"#4 #Y
    elif((color[0] > top_limit) & (color[1] < low_limit) & (color[2] > top_limit)):
        return "M"#5 #M
    elif((color[0] < low_limit) & (color[1] > top_limit) & (color[2] > top_limit)):
        return "C"#6 #C
    else:
        return "?"#0 #?

def colorline_detection(I_RGB, y, peaks):      # erkennt alle farben in einer reihe
    colorcombination = [""] * len(peaks)
    for i in range(len(peaks)):
        # mark_window(x, y_scan)
        ambient, I_RGB, y, x, color, h = color_detection(I_RGB, y, i, peaks)
        # print_values(ambient, I_rgb, y, x, color, h)
        colorcombination[i] = color_mapping(color, 0.5, 0.2)
    return colorcombination

def find_window(projector_values, colorcombination, all_peaks, window_size):
    points_camera = []
    points_projector = []
    latest_color = 0
    counter  = 0
    for i in range(len(colorcombination) - window_size + 1):
        window = tuple(colorcombination[i: i + window_size])
        points = projector_values.get(window)
        if points is not None:
            counter += 1
            for j in range(max(0, latest_color - i), len(points)):
                points_camera.append(all_peaks[i + j])
                points_projector.append(points[j])
            latest_color = i + window_size
    return points_camera, points_projector

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

def calc_multiple_lines_time(I_RGB, I_BGR, V_norm):   #Funktion zu langsam
    if one_line is False:
        all_points_camera = []
        all_points_projector = []
        
        I_RGB_max = np.max(I_RGB, axis=2)   # shape: (H, W)
        projector_values = get_projector_values(5)
        
        time_mask = 0
        time_stripes = 0
        time_true_stripes = 0
        time_window = 0
        
        start_loop = time.perf_counter()
        for y in range(0, I_RGB.shape[0]): 
            start_mask = time.perf_counter()
            intervall_anfang, intervall_ende = find_mask(V_norm, y)             #find mask
            end_mask = time.perf_counter()
            time_mask += (end_mask - start_mask)

            start_stripes = time.perf_counter()
            peaks = detect_stripes(I_RGB_max, y)                                #detect stripes
            end_stripes = time.perf_counter()
            time_stripes += (end_stripes - start_stripes)

            start_true_stripes = time.perf_counter()
            peaks = find_true_stripes(peaks,intervall_anfang, intervall_ende)   #true stripes
            end_true_stripes = time.perf_counter()
            time_true_stripes += (end_true_stripes - start_true_stripes)

            start_window = time.perf_counter()
            if len(peaks) >= 5 and len(projector_values) >= 5:
                colorcombination = colorline_detection(I_RGB, y, peaks)          #colorline detection
                c, p = find_window(projector_values, colorcombination, peaks, 5) #find window
                for i in range(len(c)):
                    all_points_camera.append((c[i],y))
                    all_points_projector.append((p[i],y))
            else:
                pass
            end_window = time.perf_counter()
            time_window += (end_window - start_window)
        
        end_loop = time.perf_counter()
        print("Time for loop     : " + str(end_loop - start_loop))
        print("-----------------------------------------------")
        print("Time for maske    : " + str(round(time_mask, 3))+ " s")
        print("Time for stripes  : " + str(round(time_stripes,3))+ " s")
        print("Time for true_str : " + str(round(time_true_stripes,3))+ " s")
        print("Time for window   : " + str(round(time_window, 3))+ " s")
        
        show_peaks = False
        if show_peaks:
            overlay = I_RGB.copy().astype(float) / 255.0
        
            plt.figure(figsize=(10,10))
            plt.imshow(overlay)
            plt.title("Intervalle und Peaks")
            plt.show()
        return all_points_camera, all_points_projector
    return None, None

def calc_multiple_lines(I_RGB, I_BGR, V_norm):   #Funktion zu langsam
    if one_line is False:
        all_points_camera = []
        all_points_projector = []
        I_RGB_max = np.max(I_RGB, axis=2)   # shape: (H, W)
        projector_values = get_projector_values(5)
        
        for y in range(0, I_RGB.shape[0]): 
            intervall_anfang, intervall_ende = find_mask(V_norm, y)             #find mask
            peaks = detect_stripes(I_RGB_max, y)                                #detect stripes
            peaks = find_true_stripes(peaks,intervall_anfang, intervall_ende)   #true stripes
            if len(peaks) >= 5 and len(projector_values) >= 5:
                colorcombination = colorline_detection(I_RGB, y, peaks)          #colorline detection
                c, p = find_window(projector_values, colorcombination, peaks, 5) #find window
                for i in range(len(c)):
                    all_points_camera.append((c[i],y))
                    all_points_projector.append((p[i],y))
        return all_points_camera, all_points_projector
    return None, None

def click_img():
    folder = os.path.dirname(os.path.abspath(__file__))
    image_path = folder + "/images/testStripes" + str(test_img_num) + ".jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Bild konnte nicht geladen werden.")
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Klicke ins Bild – Enter zum Beenden")
    cords_x = []
    cords_y = [] 
    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        x = int(event.xdata)
        y = int(event.ydata)
        print(x, y)
        cords_x.append(x)
        cords_y.append(y)
        size = 10

        # Weißes Kreuz zeichnen
        ax.plot([x - size, x + size], [y - size, y + size], color='white', linewidth=2)
        ax.plot([x - size, x + size], [y + size, y - size], color='white', linewidth=2)

        fig.canvas.draw()
    
    def on_key(event):
        if event.key == "enter":
            plt.close(fig)
    # Event registrieren
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return cords_x, cords_y

def pipeline_pixel(I_RGB, I_BGR, V_norm, x, y):
    I_RGB_max = np.max(I_RGB, axis=2)   # shape: (H, W)
    projector_values = get_projector_values(5)
    
    intervall_anfang, intervall_ende = find_mask(V_norm, y)             #find mask
    peaks = detect_stripes(I_RGB_max, y)                                #detect stripes
    #peaks = find_true_stripes(peaks,intervall_anfang, intervall_ende)   #true stripes
    
    for i in range(len(peaks)):
        if x <= peaks[i]:
            win_start = i-4
            win_end = i+4
            if(win_start < 0):
                win_start = 0
            if(win_end > len(peaks)):
                win_end = len(peaks)
            peaks = peaks[win_start: win_end]
            break
    if len(peaks) >= 5:
        colorcombination = colorline_detection(I_RGB, y, peaks)          #colorline detection
        c, p = find_window(projector_values, colorcombination, peaks, 5) #find window
        for i in range(len(c)):
            if x < c[i]:
                cam = (c[i],y)
                proj = (p[i],y)
                break
            else:
                cam = (c[len(c)-1],y)
                proj = (p[len(c)-1],y)
    else:
        return None, None
    
    if len(c) == 0:
        return None, None
    else:
        return cam, proj
#================================================================================================
# Main
#================================================================================================
def main():
    cam_pxl = []
    proj_pxl = []
    I_RGB, I_BGR, V_norm = get_vessels()
    x, y = click_img()
    for i in range(len(x)):
        c, p = pipeline_pixel(I_RGB, I_BGR, V_norm, x[i], y[i])
        if c is not None:
            cam_pxl.append(c)
            proj_pxl.append(p)
    print ("Punkte gefunden: " + str(len(proj_pxl)))
    compare_img_proj(cam_pxl, proj_pxl)
    return cam_pxl, proj_pxl

if __name__ == "__main__":
    main()
#================================================================================================
# Images and Plots
#================================================================================================
if save_img and one_line is False:
    #img_vesselnessmap(V_norm)
    #mark_stripes()
    count = 1
    for i in I_sigmas:
        #save_image(i, "gray", f"img{test_img_num}_blurred   Sigma: " + str(count), f"images/img{test_img_num}_blurred_{sigmas[count-1]}.png" )
        count += 1

if save_plot and one_line is True:
    #plot_array(I_RGB, "img" + str(test_img_num) + "_Original.png", False)
    if save_img is False:
        mark_stripes()
    #plot_array_over_img(I_sigmas, "img" + str(test_img_num) + "_Sigmas", True, I_RGB)
    # in img_labda ist ein plotaufruf für Ixx
    #plot_array_over_img(V_all, "img" + str(test_img_num) + "_V_all_img", True, I_RGB)
    plot_array_over_img(V_max, "img" + "_V_max_img", False, I_RGB, 20)
    #plot_array(V_max, "img" + str(test_img_num) + "_V_max", False)