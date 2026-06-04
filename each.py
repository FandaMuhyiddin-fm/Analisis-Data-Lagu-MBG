import pandas as pd
import numpy as np
import re
import os

# 1. Masukkan nama file dataset kamu
file_name = 'dataset_tiktok-scraper-ultimate_2026-06-04_13-01-33-459.csv'

# Membaca dataset
print("Loading raw dataset...")
df = pd.read_csv(file_name, low_memory=False)

print("Mapping columns...")
mapped_df = pd.DataFrame()

# Mapping avatar
if 'author/avatar_thumb/url_prefix' in df.columns and 'author/avatar_thumb/uri' in df.columns:
    mapped_df['authorMeta.avatar'] = df['author/avatar_thumb/url_prefix'].fillna('') + df['author/avatar_thumb/uri'].fillna('')
elif 'author/avatar_168x168/url_prefix' in df.columns and 'author/avatar_168x168/uri' in df.columns:
    mapped_df['authorMeta.avatar'] = df['author/avatar_168x168/url_prefix'].fillna('') + df['author/avatar_168x168/uri'].fillna('')
else:
    mapped_df['authorMeta.avatar'] = ''

# Mapping basic columns
mapped_df['authorMeta.name'] = df['author/unique_id'].fillna('unknown')
mapped_df['text'] = df['desc'].fillna('')
mapped_df['diggCount'] = pd.to_numeric(df['statistics/digg_count'], errors='coerce').fillna(0).astype(int)
mapped_df['shareCount'] = pd.to_numeric(df['statistics/share_count'], errors='coerce').fillna(0).astype(int)
mapped_df['playCount'] = pd.to_numeric(df['statistics/play_count'], errors='coerce').fillna(0).astype(int)
mapped_df['commentCount'] = pd.to_numeric(df['statistics/comment_count'], errors='coerce').fillna(0).astype(int)
mapped_df['collectCount'] = pd.to_numeric(df['statistics/collect_count'], errors='coerce').fillna(0).astype(int)

# Mapping duration: typically milliseconds in some datasets, let's convert to seconds if > 3600
if 'video/duration' in df.columns:
    durations = pd.to_numeric(df['video/duration'], errors='coerce').fillna(0)
    # If duration values are very large, they are likely in milliseconds
    mapped_df['videoMeta.duration'] = durations.apply(lambda x: x / 1000.0 if x > 1000 else x)
else:
    mapped_df['videoMeta.duration'] = 15.0

# Mapping music info
mapped_df['musicMeta.musicName'] = df['music/title'].fillna('Kanda My little Bolu Ketan')
mapped_df['musicMeta.musicAuthor'] = df['music/author'].fillna('VOKALIZ_NETIZEN')

if 'music/is_original' in df.columns:
    mapped_df['musicMeta.musicOriginal'] = df['music/is_original'].fillna(False).astype(bool)
else:
    mapped_df['musicMeta.musicOriginal'] = False

# Mapping date/time to createTimeISO
if 'create_time' in df.columns:
    epochs = pd.to_numeric(df['create_time'], errors='coerce')
    # Convert Unix timestamp to ISO format string
    mapped_df['createTimeISO'] = pd.to_datetime(epochs, unit='s', errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
else:
    mapped_df['createTimeISO'] = pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S.000Z')

# Mapping webVideoUrl
if 'share_url' in df.columns:
    mapped_df['webVideoUrl'] = df['share_url'].fillna('')
elif 'aweme_id' in df.columns:
    mapped_df['webVideoUrl'] = 'https://www.tiktok.com/@' + df['author/unique_id'].fillna('user') + '/video/' + df['aweme_id'].astype(str)
else:
    mapped_df['webVideoUrl'] = ''

# Filter rows where unique_id exists to ensure data quality
mapped_df = mapped_df[mapped_df['authorMeta.name'] != 'unknown']

# ===== ROBUST DEDUPLICATION =====
# Extract the numeric video ID from the TikTok URL (the true unique key)
def extract_vid_id(url):
    if not isinstance(url, str):
        return ''
    m = re.search(r'/video/(\d+)', url)
    return m.group(1) if m else ''

mapped_df['vid_id'] = mapped_df['webVideoUrl'].apply(extract_vid_id)

# Clean URLs: strip query parameters for display
def clean_url(url):
    if not isinstance(url, str):
        return url
    return url.split('?')[0]

mapped_df['webVideoUrl'] = mapped_df['webVideoUrl'].apply(clean_url)

# Sort by playCount descending so we keep the row with the highest stats
mapped_df = mapped_df.sort_values(by='playCount', ascending=False)

# Deduplicate: keep first (highest playCount) per unique video ID
before_dedup = len(mapped_df)
mapped_df = mapped_df.drop_duplicates(subset=['vid_id'], keep='first')

# Also deduplicate by caption + author as a secondary safety net
# (catches cases where vid_id extraction failed but same content exists)
mapped_df = mapped_df.drop_duplicates(subset=['text', 'authorMeta.name', 'createTimeISO'], keep='first')

# Drop the helper column
mapped_df = mapped_df.drop(columns=['vid_id'])

after_dedup = len(mapped_df)
print(f"Deduplicated: {before_dedup} -> {after_dedup} rows (removed {before_dedup - after_dedup} duplicates)")

# Make sure directory data/ exists
os.makedirs('data', exist_ok=True)

output_path = os.path.join('data', 'dataset_video.csv')
mapped_df.to_csv(output_path, index=False)

print(f"Selesai! Data berhasil diekstrak dan disimpan di: {output_path}")
print(f"Jumlah baris hasil ekstraksi: {len(mapped_df)}")