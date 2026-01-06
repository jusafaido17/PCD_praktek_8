import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import OrderedDict # Untuk menjaga urutan hasil

# --- KONFIGURASI UTAMA ---
FILENAME = 'phantom.png' 
SPATIAL_RESOLUTION = 1.4798 
MIN_OBJECT_AREA = 100 # Sesuaikan jika objek hilang atau terlalu banyak noise
MANUAL_THRESHOLD = 150 # Nilai Threshold manual untuk citra phantom yang sulit di-Otsu
# -------------------------

def hitung_jarak_geometri(filename, res):
    
    img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    if img_gray is None:
        print(f"❌ ERROR: Tidak dapat memuat file '{filename}'.")
        return

    # 1. Thresholding Citra (Gunakan MANUAL_THRESHOLD untuk mengatasi kegagalan Otsu's)
    # Ubah MANUAL_THRESHOLD jika objek masih menyatu
    _, img_binary = cv2.threshold(img_gray, MANUAL_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # 2. Labelling Objek & Menentukan Centroid
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_centroids = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_OBJECT_AREA:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                all_centroids.append({'cx': cx, 'cy': cy, 'area': area})

    # PENTING: SORTING BERDASARKAN POSISI SPASIAL (cy, cx)
    # Ini menentukan urutan Objek 1, 2, 3, 4 sesuai skema modul (dari kiri-atas ke kanan-bawah)
    sorted_centroids = sorted(all_centroids, key=lambda p: (p['cy'], p['cx']))
    
    if len(sorted_centroids) < 4:
        print(f"❌ ERROR: Hanya ditemukan {len(sorted_centroids)} objek yang lulus filter.")
        print(f"Pastikan Plot 2 menunjukkan 4 objek putih terpisah, atau ganti THRESHOLD/MIN_OBJECT_AREA.")
        return
        
    # Ambil 4 objek pertama sesuai urutan spasial
    centroids_map = OrderedDict([
        (1, sorted_centroids[0]), # Objek 1
        (2, sorted_centroids[1]), # Objek 2
        (3, sorted_centroids[2]), # Objek 3
        (4, sorted_centroids[3])  # Objek 4
    ])
    
    # --- FUNGSI PENGUKURAN JARAK ---
    def euclidean_distance(p1, p2):
        return sqrt((p1['cx'] - p2['cx'])**2 + (p1['cy'] - p2['cy'])**2)
    
    # Semua pasangan yang diminta (6 kombinasi)
    pairs_to_measure = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    measurements = []
    
    # PENGUKURAN JARAK EUCLIDEAN
    for i, j in pairs_to_measure:
        p_i = centroids_map[i]
        p_j = centroids_map[j]
        
        d_px = euclidean_distance(p_i, p_j)
        d_mm = d_px / res
        
        measurements.append({
            'pair': (i, j),
            'd_px': d_px,
            'd_mm': d_mm,
            'p1': p_i,
            'p2': p_j
        })

    # --- VISUALISASI DAN OUTPUT ---
    img_result = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    print("\n=========================================================")
    print(f"HASIL PENGUKURAN POLA GEOMETRI PADA '{filename}'")
    print(f"Threshold Manual: {MANUAL_THRESHOLD}")
    print("=========================================================")

    # Labeling Centroid
    for key, c in centroids_map.items():
        cv2.circle(img_result, (c['cx'], c['cy']), 8, (0, 255, 0), -1)
        cv2.putText(img_result, f'{key}', (c['cx'] + 10, c['cy'] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    # Menggambar garis dan mencetak hasil
    line_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    
    for idx, measure in enumerate(measurements):
        p1 = measure['p1']
        p2 = measure['p2']
        color = line_colors[idx % len(line_colors)]
        
        # Gambar garis jarak
        cv2.line(img_result, (p1['cx'], p1['cy']), (p2['cx'], p2['cy']), color, 2)
        
        # Tulis hasil ke konsol
        print(f"{idx+1}. Jarak Objek {measure['pair'][0]} dan Objek {measure['pair'][1]}:")
        print(f"   -> Jarak Piksel (d_px): {measure['d_px']:.4f}")
        print(f"   -> Jarak Milimeter (d_mm): {measure['d_mm']:.4f}")
        print("---------------------------------------------------------")

    # Menampilkan Citra Hasil (3 Plot: Grayscale, Biner, Hasil)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Praktek 8: Pengukuran Pola Geometri ({len(measurements)} Jarak)", fontsize=16)

    # Plot 1: Grayscale
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title("1. Citra Grayscale")
    axes[0].axis('off')
    
    # Plot 2: Citra Biner (DIAGNOSTIK: Cek pemisahan objek)
    axes[1].imshow(img_binary, cmap='gray')
    axes[1].set_title(f"2. Citra Biner (Threshold={MANUAL_THRESHOLD})")
    axes[1].axis('off')
    

    # Plot 3: Citra Hasil dengan Jarak
    axes[2].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    axes[2].set_title("3. Hasil Pengukuran Jarak Centroid")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show(block=True) 

# --- JALANKAN PROGRAM PRAKTEK 8 (Pola Geometri) ---
print(f"Memulai Ekstraksi Ciri Pola Geometri pada '{FILENAME}'...")
hitung_jarak_geometri(FILENAME, SPATIAL_RESOLUTION)