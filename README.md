# RiceLeaf AI

Web klasifikasi penyakit daun padi untuk kelas **Blast**, **Blight**, dan
**Tungro**. Model Random Forest menggunakan 42 fitur yang mencakup histogram
HSV, bentuk daun, area penyakit, tekstur, dan Canny edge.

## Menjalankan secara lokal

Gunakan Python 3.11 atau 3.12.

```bash
python -m venv .venv
```

Aktifkan virtual environment, kemudian jalankan:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Struktur file

- `app.py`: aplikasi dan preprocessing citra.
- `riceleaf.joblib`: model Random Forest beserta metadata.
- `requirements.txt`: versi dependensi yang sesuai dengan model.
- `.streamlit/config.toml`: konfigurasi tampilan dan server.

## Publikasi

Paket ini dapat dipublikasikan melalui Streamlit Community Cloud. Simpan
seluruh isi folder dalam satu repository GitHub, pilih `app.py` sebagai main
file, lalu deploy.

> Hasil aplikasi merupakan prediksi model dan bukan pengganti pemeriksaan ahli
> pertanian.
