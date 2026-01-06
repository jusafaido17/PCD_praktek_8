import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# --- KONFIGURASI ---
FILENAME = 'Finding_Nemo02.jpg' 
DISTANCES = [1, 2, 3] 
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4] 
# -------------------

def analisis_tekstur_dipisah(filename):
    img = cv2.imread(filename)
    if img is None:
        print(f"❌ ERROR: File '{filename}' tidak ditemukan.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for d in DISTANCES:
        # 1. Hitung GLCM
        glcm = graycomatrix(gray, [d], ANGLES, levels=256, symmetric=True, normed=True)

        # 2. Ekstraksi Fitur
        contrast = graycoprops(glcm, 'contrast')[0]
        correlation = graycoprops(glcm, 'correlation')[0]
        energy = graycoprops(glcm, 'energy')[0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0]

        # 3. Tampilkan Tabel
        print(f"\n" + "="*80)
        print(f" HASIL GLCM - PIXEL DISTANCE = {d} ".center(80, " "))
        print("="*80)
        print(f"{'Fitur':<15} | {'0°':<10} | {'45°':<10} | {'90°':<10} | {'135°':<10} | {'Rata-rata':<10}")
        print("-" * 80)

        features = {
            "Contrast": contrast,
            "Correlation": correlation,
            "Energy": energy,
            "Homogeneity": homogeneity
        }

        for name, values in features.items():
            avg = np.mean(values)
            print(f"{name:<15} | {values[0]:<10.4f} | {values[1]:<10.4f} | {values[2]:<10.4f} | {values[3]:<10.4f} | {avg:<10.4f}")
        print("-" * 80)

        # 4. Visualisasi (Perbaikan bagian ini)
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(gray, cmap='gray')
        plt.title(f"Citra Input (Jarak: {d})")
        plt.axis('off')

        # Visualisasi Matriks GLCM (Ambil sudut 0 derajat sebagai perwakilan)
        # Struktur glcm adalah [i, j, d, a]
        glcm_vis = glcm[:, :, 0, 0] 
        
        plt.subplot(1, 2, 2)
        # Gunakan log1p untuk memperjelas visualisasi probabilitas yang kecil
        plt.imshow(np.log1p(glcm_vis), cmap='viridis')
        plt.title(f"Matriks GLCM Sudut 0° (Jarak: {d})")
        plt.colorbar(label='Log intensity')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print(f"Memulai Analisis Tekstur pada '{FILENAME}'...")
    analisis_tekstur_dipisah(FILENAME)