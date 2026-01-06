import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt

# --- KONFIGURASI UTAMA ---
FILENAME = 'pillsetc.png' 
MIN_AREA = 30 # Luas minimum objek yang dipertahankan (sesuai bwareaopen(bw, 30))
R_CLOSE = 2 # Radius Strel untuk operasi Closing
# -------------------------

def get_classification(metric, eccentricity):
    """
    Mengklasifikasikan bentuk objek menggunakan aturan if sederhana.
    Berdasarkan Metric: Metric mendekati 1 = Bulat, mendekati 0 = Memanjang.
    """
    if metric > 0.8:
        return "Bulat (Round)"
    elif eccentricity > 0.85: # Jika Metric agak rendah, cek Eccentricity
        return "Memanjang (Elongated)"
    else:
        return "Bentuk Lain (Other)"


def hitung_ciri_bentuk(filename):
    
    # 1. Membaca citra RGB
    img_bgr = cv2.imread(filename)
    if img_bgr is None:
        print(f"❌ ERROR: Tidak dapat memuat file '{filename}'. Pastikan file ada.")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Mengkonversi citra RGB menjadi citra grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 3. Mengkonversi citra grayscale menjadi citra biner menggunakan Otsu
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # --- PROSES CLEANING MORFOLOGI SESUAI URUTAN MODUL ---
    
    # 4. Menghilangkan noise dengan cara menghapus objek yang memiliki luas di bawah 30 (bwareaopen)
    # Langkah ini dilakukan dengan mencari kontur awal, memfilter, dan menggambar kontur yang lolos
    initial_contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_bwareaopen = np.zeros_like(img_binary)
    
    # Memfilter dan menggambar kontur yang luasnya >= MIN_AREA
    for c in initial_contours:
        if cv2.contourArea(c) >= MIN_AREA:
            cv2.drawContours(img_bwareaopen, [c], -1, 255, -1) # Isi kontur putih
            
    # 5. Operasi morfologi yaitu closing dan filling holes
    
    # A. Closing (imclose: Strel Disk, R=2)
    se_size = 2 * R_CLOSE + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    img_closed = cv2.morphologyEx(img_bwareaopen, cv2.MORPH_CLOSE, kernel)
    
    # B. Filling Holes (imfill)
    # Kita menggunakan findContours dengan mode CCOMP untuk identifikasi lubang, lalu mengisi lubang
    contours_hole, hierarchy = cv2.findContours(img_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    img_filled = img_closed.copy()
    
    if hierarchy is not None:
        for i in range(len(contours_hole)):
            # Jika kontur adalah lubang (memiliki parent), isi dengan putih
            if hierarchy[0][i][3] != -1: 
                cv2.drawContours(img_filled, contours_hole, i, 255, -1)

    # 6. Labelling objek & 7. Menghitung ciri (Luas, Keliling, Metric, Eccentricity)
    
    # Cari final kontur (objek luar) pada citra yang sudah bersih dan terisi
    final_contours, _ = cv2.findContours(img_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_result = img_rgb.copy()
    results = []
    
    for k, contour in enumerate(final_contours):
        
        # Hitung Ciri Ukuran
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Pastikan kontur cukup besar untuk fitEllipse dan perhitungan
        if len(contour) < 5 or perimeter == 0:
            continue
            
        # Hitung Ciri Bentuk: Eccentricity & Metric
        
        # Eccentricity (dari Elips yang paling cocok)
        try:
            (center, axes, orientation) = cv2.fitEllipse(contour)
            major_axis = max(axes) / 2
            minor_axis = min(axes) / 2
            eccentricity = sqrt(1 - (minor_axis**2 / major_axis**2))
            
            # Centroid (Pusat Massa)
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        except Exception:
            eccentricity = 1.0 
            cx, cy = 0, 0
            
        # Metric
        metric = (4 * pi * area) / (perimeter ** 2)
        
        # 8. Klasifikasikan bentuk objek ('aturan if' sederhana)
        classification = get_classification(metric, eccentricity)
        
        # Simpan hasil
        results.append({
            'k': k + 1,
            'area': area,
            'perimeter': perimeter,
            'eccentricity': eccentricity,
            'metric': metric,
            'centroid': (cx, cy),
            'classification': classification
        })

        # --- VISUALISASI PADA CITRA HASIL ---
        cv2.drawContours(img_result, [contour], -1, (255, 255, 255), 2)
        cv2.putText(img_result, str(k + 1), (cx, cy - 16), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
    # --- CETAK HASIL DI KONSOL ---
    print("=========================================================")
    print(f"HASIL EKSTRAKSI CIRI BENTUK PADA '{filename}' ({len(results)} OBJEK)")
    print("=========================================================")
    for res in results:
        print(f"Objek ke-{res['k']}: ({res['classification']})")
        print(f"  Luas (Area):        {res['area']:.2f} piksel")
        print(f"  Keliling (Perimeter): {res['perimeter']:.2f} piksel")
        print(f"  Eccentricity:     {res['eccentricity']:.4f}")
        print(f"  Metric:           {res['metric']:.4f}")
        print("---------------------------------------------------------")


    # 9. MENAMPILKAN CITRA HASIL
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Praktek 8: Ekstraksi Pola Bentuk", fontsize=16)

    # Plot 1: Citra Grayscale
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title("1. Citra Grayscale")
    axes[0].axis('off')
    
    # Plot 2: Citra Biner (Setelah Cleaning Morfologi)
    axes[1].imshow(img_filled, cmap='gray')
    axes[1].set_title("2. Citra Biner Bersih (Setelah Bwareaopen, Closing, Imfill)")
    axes[1].axis('off')
    
    # Plot 3: Citra Hasil dengan Kontur dan Label
    axes[2].imshow(img_result)
    axes[2].set_title(f"3. Hasil Ekstraksi Kontur ({len(results)} Objek)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show(block=True) 

# --- JALANKAN PROGRAM PRAKTEK 8 ---
print(f"Memulai Ekstraksi Ciri Pola Bentuk pada '{FILENAME}'...")
hitung_ciri_bentuk(FILENAME)