import pandas as pd

# File input CSV
INPUT_FILE = "dataset_tiktok-comment-scraper_2026-06-04_16-24-20-815.csv"

# File output
OUTPUT_FILE = "dataset_komentar_convert.csv"

# Baca CSV
df = pd.read_csv(INPUT_FILE)

# Mapping kolom
result = pd.DataFrame({
    "text": df["text"],
    "username": df["user/unique_id"],
    "likeCount": df["digg_count"],
    "replyCount": df["reply_comment_total"],
    "isReply": False,
    "parentCommentId": None,
    "createTime": pd.to_datetime(
        df["create_time"],
        unit="s",
        errors="coerce"
    ).dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    "videoUrl": "https://www.tiktok.com/video/" + df["aweme_id"].astype(str)
})

# Hapus komentar kosong
result = result.dropna(subset=["text"])

# Simpan ke CSV baru
result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"Berhasil disimpan ke {OUTPUT_FILE}")
print(result.head())