# RiceLeaf AI — Streamlit

Web identifikasi dan visualisasi empat penyakit tanaman padi:

- Bacterial Blight
- Blast
- Brown Spot
- Tungro

## 1. Masukkan model

Unduh file berikut dari Google Drive:

`/content/drive/MyDrive/RiceLeaf/runs/experiment_v1/riceleaf_bundle.zip`

Ekstrak arsip tersebut, lalu salin tiga file berikut ke folder `model/`:

```text
model/
├── classifier.pt
├── segmenter.pt
└── metadata.json
```

## 2. Jalankan secara lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3. Deploy melalui Streamlit Community Cloud

1. Unggah seluruh isi folder ini ke repository GitHub.
2. Pastikan tiga file model terlihat di folder `model/`.
3. Buka Streamlit Community Cloud dan pilih **Deploy a public app from GitHub**.
4. Isi repository dan branch `main`.
5. Isi **Main file path** dengan `app.py`.
6. Tekan **Deploy**.

## Batasan penting

Model saat ini belum memiliki kelas daun padi normal dan belum mempunyai detektor input bukan padi. Gambar seperti itu dapat tetap dipaksa masuk ke salah satu dari empat penyakit. Web menampilkan peringatan ini kepada pengguna.
