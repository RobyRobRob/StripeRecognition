import cupy as cp
from cupyx.scipy.signal import find_peaks

# Beispieldaten auf GPU
x_gpu = cp.array([[0, 1, 3, 7, 5, 2, 6, 8, 4, 1],[0, 1, 3, 7, 5, 2, 6, 8, 4, 1]])

peaks_gpu, properties = find_peaks(
    x_gpu,
    prominence=1,
    distance=2
)

print(peaks_gpu)        # GPU Array
print(properties)

peaks = peaks_gpu.get()