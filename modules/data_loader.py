import os
import re
import pandas as pd
import streamlit as st

def extract_video_id(url):
    """
    Extracts the unique numeric video ID from a TikTok URL.
    Example: https://www.tiktok.com/@user/video/7647083637132823815 -> 7647083637132823815
    """
    if not isinstance(url, str):
        return ""
    match = re.search(r'/video/(\d+)', url)
    return match.group(1) if match else ""

def find_file(filename):
    """
    Finds a file in common locations: root, data/, or parent folder.
    """
    possible_paths = [
        filename,
        os.path.join("data", filename),
        os.path.join("..", filename),
        os.path.join("..", "data", filename)
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return filename

@st.cache_data
def load_video_data():
    """
    Loads and cleans dataset_video.csv.
    """
    path = find_file("dataset_video.csv")
    if not os.path.exists(path):
        st.error(f"File {path} tidak ditemukan!")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path)
        
        # Clean up column names by stripping whitespace
        df.columns = df.columns.str.strip()
        
        # Ensure critical columns exist and are of correct type
        numeric_cols = [
            "diggCount", "shareCount", "playCount", 
            "commentCount", "collectCount", "videoMeta.duration"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = 0.0
                
        # Parse datetime
        if "createTimeISO" in df.columns:
            df["parsedTime"] = pd.to_datetime(df["createTimeISO"], errors="coerce")
        else:
            df["parsedTime"] = pd.Timestamp.now()
            
        # Extract video ID for robust matching with comments
        if "webVideoUrl" in df.columns:
            # Clean URLs: strip query parameters
            df["webVideoUrl"] = df["webVideoUrl"].apply(
                lambda u: u.split('?')[0] if isinstance(u, str) else u
            )
            df["videoId"] = df["webVideoUrl"].apply(extract_video_id)
        else:
            df["videoId"] = ""
            df["webVideoUrl"] = ""
            
        # Rename display column headers where necessary
        df["author_name"] = df["authorMeta.name"].fillna("unknown")
        df["caption"] = df["text"].fillna("")
        
        # ===== ROBUST DEDUPLICATION =====
        # Sort by playCount descending to keep the row with highest stats
        df = df.sort_values(by="playCount", ascending=False)
        
        # Primary dedup: by numeric video ID (the true unique key)
        if "videoId" in df.columns:
            df_with_id = df[df["videoId"] != ""]
            df_no_id = df[df["videoId"] == ""]
            df_with_id = df_with_id.drop_duplicates(subset=["videoId"], keep="first")
            df = pd.concat([df_with_id, df_no_id], ignore_index=True)
        
        # Secondary dedup: by caption + author + time (catches edge cases)
        dedup_cols = ["caption", "author_name", "createTimeISO"]
        existing_dedup = [c for c in dedup_cols if c in df.columns]
        if existing_dedup:
            df = df.drop_duplicates(subset=existing_dedup, keep="first")
            
        return df
    except Exception as e:
        st.error(f"Error loading video data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_comment_data():
    """
    Loads and cleans dataset_komentar.csv.
    Optimized: First tries to load pre-classified comments.
    """
    # Look for sentiment-precomputed comments first for maximum speed
    path = find_file("dataset_komentar_sentiment.csv")
    if not os.path.exists(path):
        path = find_file("dataset_komentar.csv")
        
    if not os.path.exists(path):
        st.error(f"File komentar tidak ditemukan!")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path)
        
        # Clean up columns
        df.columns = df.columns.str.strip()
        
        # Clean numeric
        numeric_cols = ["likeCount", "replyCount"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = 0
                
        # Parse timestamp
        if "createTime" in df.columns:
            df["parsedTime"] = pd.to_datetime(df["createTime"], errors="coerce")
        else:
            df["parsedTime"] = pd.Timestamp.now()
            
        # Extract video ID for matching
        if "videoUrl" in df.columns:
            df["videoId"] = df["videoUrl"].apply(extract_video_id)
        else:
            df["videoId"] = ""
            df["videoUrl"] = ""
            
        df["comment_text"] = df["text"].fillna("")
        df["username"] = df["username"].fillna("anonymous")
        
        # Remove duplicate comments
        subset_cols = ["text", "username", "createTime"]
        existing_subset = [c for c in subset_cols if c in df.columns]
        if existing_subset:
            df = df.drop_duplicates(subset=existing_subset)
            
        return df
    except Exception as e:
        st.error(f"Error loading comment data: {e}")
        return pd.DataFrame()
