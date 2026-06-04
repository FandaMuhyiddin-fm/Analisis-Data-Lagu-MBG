# 📊 Dashboard Analisa Sentimen TikTok

Dashboard interaktif berbasis Python + Streamlit untuk menganalisa engagement video TikTok dan sentimen komentar, dengan ringkasan AI menggunakan Claude API.

---

## 🚀 Cara Setup & Menjalankan

### 1. Clone / download project

```bash
git clone <your-repo-url>
cd sentiment_dashboard
```

### 2. Buat virtual environment (disarankan)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Catatan:** Instalasi `torch` dan `transformers` (IndoBERT) membutuhkan beberapa menit dan ~2GB storage. Jika ingin versi ringan, hapus dua baris tersebut dari `requirements.txt` — dashboard akan otomatis menggunakan VADER sebagai fallback.

### 4. Konfigurasi API Key

```bash
cp .env.example .env
# Edit .env dan isi ANTHROPIC_API_KEY dengan API key Anda
```

Atau, masukkan API key langsung di sidebar saat dashboard berjalan.

### 5. Jalankan dashboard

```bash
streamlit run app.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`

---

## 📁 Struktur Project

```
sentiment_dashboard/
│
├── app.py                      # Entry point Streamlit
│
├── modules/
│   ├── __init__.py
│   ├── data_loader.py          # Load & validasi CSV
│   ├── engagement.py           # Hitung & tampilkan engagement score
│   ├── sentiment.py            # Analisa sentimen komentar (IndoBERT/VADER)
│   ├── first_post.py           # Tabel postingan pertama dengan filter
│   └── ai_summary.py           # Generate ringkasan via Claude API
│
├── utils/
│   ├── __init__.py
│   └── helpers.py              # Fungsi utilitas (format angka, tanggal, dll)
│
├── data/                       # Simpan file CSV di sini (tidak di-commit)
│   ├── dataset_video.csv
│   └── dataset_komentar.csv
│
├── output/                     # Hasil ekspor (tidak di-commit)
│
├── requirements.txt
├── .env.example
├── .env                        # TIDAK di-commit!
└── .gitignore
```

---

## 📊 Fitur Dashboard

### Tab 1 — 🔥 Engagement Video
- Metrik ringkasan (total video, play, like, komentar)
- Kartu Top 3 Video dengan badge medal
- Bar chart Top 20 video berdasarkan engagement score
- Scatter plot: Play Count vs Like Count (ukuran = jumlah komentar)
- Tabel lengkap semua video yang dapat diurutkan

### Tab 2 — 💬 Analisa Sentimen
- Distribusi sentimen (Positif / Netral / Negatif) per video via donut chart
- Word cloud komentar per video
- Statistik sentimen dengan persentase
- Tabel komentar dengan filter berdasarkan label sentimen

### Tab 3 — 📅 Postingan Pertama
- Tabel diurutkan dari postingan terlama ke terbaru
- Filter berdasarkan akun (multi-select)
- Filter berdasarkan rentang tanggal
- Pencarian teks bebas pada caption
- Link langsung ke setiap video

### Tab 4 — 🤖 AI Summary
- Preview data context yang dikirim ke Claude
- Generate ringkasan naratif on-demand
- Ekspor hasil ke file `.txt`

---

## 🔧 Formula Engagement Score

```
Score = (like × 1.0) + (komentar × 3.0) + (share × 4.0) + (koleksi × 2.0) + (play × 0.1)
```

---

## 🤖 Model Sentimen

Dashboard secara otomatis memilih model terbaik yang tersedia:

1. **IndoBERT** (`mdhugol/indonesia-bert-sentiment-classification`) — akurasi tinggi untuk Bahasa Indonesia
2. **VADER** — cepat, ringan, cocok untuk Bahasa Inggris
3. **Rule-based** — fallback sederhana berbasis kata kunci

---

## ❓ Troubleshooting

| Masalah | Solusi |
|---|---|
| `torch` gagal install | Install versi CPU: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| API Key tidak bekerja | Pastikan format `sk-ant-...` dan akun Anthropic aktif |
| Komentar tidak muncul di video | Pastikan kolom `videoUrl` di komentar mengandung ID video yang sama dengan `webVideoUrl` di video |
| Dashboard lambat | Model IndoBERT berjalan di CPU; gunakan VADER untuk performa lebih cepat |
