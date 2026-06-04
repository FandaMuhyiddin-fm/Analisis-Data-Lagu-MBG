import streamlit as st
import pandas as pd
from utils.helpers import format_number, format_duration

def render_first_post_tab(df):
    """
    Renders Tab 3: Postingan Pertama details.
    Sorts videos from oldest to newest with multiple interactive filters.
    """
    if df.empty:
        st.warning("Tidak ada data video untuk ditampilkan.")
        return

    st.markdown("### Postingan Pertama & Riwayat Konten")
    st.markdown("Halaman ini menyajikan urutan postingan dari yang terlama ke terbaru. Gunakan filter di bawah ini untuk mencari postingan spesifik.")

    # Sort data by datetime (oldest first)
    sorted_df = df.sort_values(by="parsedTime", ascending=True).copy()

    # Filter section
    col1, col2 = st.columns(2)
    
    with col1:
        # Account multi-select filter
        unique_authors = sorted(sorted_df["author_name"].unique())
        selected_authors = st.multiselect(
            "Filter Akun / Kreator:",
            options=unique_authors,
            default=unique_authors,
            help="Pilih satu atau beberapa akun TikTok untuk memfilter hasil."
        )

    with col2:
        # Date range filter
        min_date = sorted_df["parsedTime"].min().date() if not sorted_df.empty else pd.Timestamp.now().date()
        max_date = sorted_df["parsedTime"].max().date() if not sorted_df.empty else pd.Timestamp.now().date()
        
        selected_dates = st.date_input(
            "Rentang Tanggal Posting:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Filter video berdasarkan rentang tanggal publish."
        )

    # Free text caption search
    search_query = st.text_input("🔍 Cari Caption / Kata Kunci:", "", placeholder="Masukkan teks pencarian di sini...")

    # Apply filters
    filtered_df = sorted_df.copy()
    
    # Filter by authors
    if selected_authors:
        filtered_df = filtered_df[filtered_df["author_name"].isin(selected_authors)]
        
    # Filter by dates
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
        filtered_df = filtered_df[
            (filtered_df["parsedTime"].dt.date >= start_date) & 
            (filtered_df["parsedTime"].dt.date <= end_date)
        ]
        
    # Filter by text search
    if search_query:
        filtered_df = filtered_df[filtered_df["caption"].str.contains(search_query, case=False, na=False)]

    # Result summary
    st.markdown(f"Menampilkan **{len(filtered_df)}** dari **{len(sorted_df)}** video.")

    # Render table
    if filtered_df.empty:
        st.info("Tidak ada data video yang cocok dengan filter yang dipilih.")
        return

    # Clean display columns
    display_df = filtered_df[[
        "author_name", "parsedTime", "caption", "playCount", 
        "diggCount", "commentCount", "videoMeta.duration", "webVideoUrl"
    ]].copy()
    
    display_df.rename(columns={
        "author_name": "Username",
        "parsedTime": "Tanggal",
        "caption": "Caption",
        "playCount": "Plays",
        "diggCount": "Likes",
        "commentCount": "Comments",
        "videoMeta.duration": "Durasi",
        "webVideoUrl": "Link Video"
    }, inplace=True)

    # Format durasi and other display aspects
    st.dataframe(
        display_df,
        column_config={
            "Link Video": st.column_config.LinkColumn("Link Video", display_text="Tonton Video ↗"),
            "Tanggal": st.column_config.DatetimeColumn("Tanggal Posting", format="YYYY-MM-DD HH:mm"),
            "Plays": st.column_config.NumberColumn("Plays", format="%d"),
            "Likes": st.column_config.NumberColumn("Likes", format="%d"),
            "Comments": st.column_config.NumberColumn("Comments", format="%d"),
            "Durasi": st.column_config.NumberColumn("Durasi (s)", format="%.1f")
        },
        use_container_width=True,
        hide_index=True
    )
