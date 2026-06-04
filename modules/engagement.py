import streamlit as st
import pandas as pd
import plotly.express as px
from utils.helpers import format_number, format_duration

def calculate_engagement_score(df):
    """
    Tidak menggunakan formula engagement score buatan lagi.
    Langsung menggunakan commentCount (Jumlah Komentar) sebagai metrik utama.
    """
    if df.empty:
        return df
    df["engagement_score"] = df["commentCount"]
    return df

def render_engagement_tab(df):
    """
    Renders Tab 1: Video details sorted strictly by comments (komentar tertinggi).
    """
    if df.empty:
        st.warning("Tidak ada data video untuk ditampilkan.")
        return

    # Ensure engagement_score matches commentCount for compatibility
    df = calculate_engagement_score(df)

    # 1. Metrik Ringkasan
    total_videos = len(df)
    total_plays = df["playCount"].sum()
    total_likes = df["diggCount"].sum()
    total_comments = df["commentCount"].sum()

    st.markdown("### Ringkasan Performa Komentar & Interaksi")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Video Unik", format_number(total_videos))
    with col2:
        st.metric("Total Komentar", format_number(total_comments))
    with col3:
        st.metric("Total Likes", format_number(total_likes))
    with col4:
        st.metric("Total Tayangan (Plays)", format_number(total_plays))

    st.markdown("---")

    # 2. Top 3 Profil Kreator Terpopuler berdasarkan Komentar Terbanyak
    st.markdown("### Top 3 Kreator Terpopuler (Komentar Terbanyak)")
    st.markdown("Daftar 3 kreator teratas dengan akumulasi komentar terbanyak pada konten mereka terkait lagu **Mas Bahlil Ganteng** (MBG).")

    # Aggregate stats per creator
    creator_stats = df.groupby("author_name").agg({
        "playCount": "sum",
        "diggCount": "sum",
        "commentCount": "sum",
        "shareCount": "sum",
        "authorMeta.avatar": "first",
        "webVideoUrl": "first"
    }).reset_index()

    # Count videos per creator
    video_counts = df.groupby("author_name").size().reset_index(name="video_count")
    creator_stats = creator_stats.merge(video_counts, on="author_name", how="left")

    top_creators = creator_stats.sort_values(by="commentCount", ascending=False).head(3)

    creator_medals = [
        {"rank_color": "#FFD700", "border": "#FFD700", "bg_gradient": "linear-gradient(135deg, rgba(255,215,0,0.15) 0%, rgba(255,165,0,0.05) 100%)", "icon": "🥇"},
        {"rank_color": "#C0C0C0", "border": "#C0C0C0", "bg_gradient": "linear-gradient(135deg, rgba(192,192,192,0.15) 0%, rgba(128,128,128,0.05) 100%)", "icon": "🥈"},
        {"rank_color": "#CD7F32", "border": "#CD7F32", "bg_gradient": "linear-gradient(135deg, rgba(205,127,50,0.15) 0%, rgba(139,69,19,0.05) 100%)", "icon": "🥉"}
    ]

    creator_cols = st.columns(3)
    for idx, (_, row) in enumerate(top_creators.iterrows()):
        avatar_url = row["authorMeta.avatar"] if pd.notna(row["authorMeta.avatar"]) and row["authorMeta.avatar"] != "" else "https://www.w3schools.com/howto/img_avatar.png"
        medal = creator_medals[idx]
        vid_url = row["webVideoUrl"] if pd.notna(row["webVideoUrl"]) else "#"
        vid_count = int(row["video_count"]) if pd.notna(row["video_count"]) else 0
        with creator_cols[idx]:
            st.markdown(
                f"""
                <div style="
                    background: {medal['bg_gradient']};
                    border: 2px solid {medal['border']};
                    border-radius: 16px;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 6px 25px rgba(0, 0, 0, 0.3);
                    height: 100%;
                    position: relative;
                ">
                    <div style="font-size: 2.2em; margin-bottom: 4px;">{medal['icon']}</div>
                    <div style="font-size: 0.75em; text-transform: uppercase; letter-spacing: 1px; color: {medal['rank_color']}; font-weight: 800; margin-bottom: 10px;">Rank #{idx+1}</div>
                    <div style="font-weight: bold; font-size: 1.1em; margin-bottom: 4px; color: #e2e8f0; word-break: break-all;">@{row['author_name']}</div>
                    <div style="font-size: 0.75em; color: #64748b; margin-bottom: 10px;">{vid_count} video</div>
                    <hr style="margin: 8px 0; border: 0.5px solid rgba(255, 255, 255, 0.1);" />
                    <div style="font-size: 0.85em; text-align: left; color: #94a3b8; line-height: 1.8; padding: 0 5px;">
                        <b>Total Komentar:</b> <span style="color: #00f2fe; font-weight: 800; font-size: 1.1em;">{format_number(row['commentCount'])}</span><br/>
                        <b>Total Likes:</b> {format_number(row['diggCount'])}<br/>
                        <b>Total Views:</b> {format_number(row['playCount'])}<br/>
                        <b>Total Shares:</b> {format_number(row['shareCount'])}
                    </div>
                    <div style="margin-top: 12px; text-align: center; padding-top: 8px;">
                        <a href="{vid_url}" target="_blank" style="text-decoration: none; color: #00f2fe; font-size: 0.85em; font-weight: bold;">Lihat Video \u2197</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # 3. Grafik Postingan dan Akun Berdasarkan Komentar Terbanyak
    st.markdown("### Analisis Komentar Terbanyak")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Card container for Video Postingan
        st.markdown(
            """
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 15px;">
                <h4 style="margin:0 0 15px 0; color: #e2e8f0;">Top 10 Video Postingan (Komentar Terbanyak)</h4>
            </div>
            """, unsafe_allow_html=True
        )
        top_videos = df.sort_values(by="commentCount", ascending=False).head(10).copy()
        top_videos["label"] = top_videos.apply(lambda r: f"@{r['author_name']} ({format_number(r['commentCount'])})", axis=1)
        
        fig_videos = px.bar(
            top_videos,
            x="commentCount",
            y="label",
            orientation="h",
            color="commentCount",
            color_continuous_scale="Reds",
            labels={"commentCount": "Jumlah Komentar", "label": "Kreator/Video"},
            template="plotly_dark"
        )
        fig_videos.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_videos, use_container_width=True)

    with col_chart2:
        # Card container for Akun Terpopuler
        st.markdown(
            """
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 15px;">
                <h4 style="margin:0 0 15px 0; color: #e2e8f0;">Top 10 Akun Kreator (Total Komentar Akumulasi)</h4>
            </div>
            """, unsafe_allow_html=True
        )
        top_accounts = creator_stats.sort_values(by="commentCount", ascending=False).head(10).copy()
        top_accounts["label"] = top_accounts.apply(lambda r: f"@{r['author_name']} ({format_number(r['commentCount'])})", axis=1)
        
        fig_accounts = px.bar(
            top_accounts,
            x="commentCount",
            y="label",
            orientation="h",
            color="commentCount",
            color_continuous_scale="Tealgrn",
            labels={"commentCount": "Total Komentar", "label": "Akun Kreator"},
            template="plotly_dark"
        )
        fig_accounts.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_accounts, use_container_width=True)

    st.markdown("---")

    # 4. Tabel Lengkap Semua Video
    st.markdown("### Daftar Semua Video")
    
    # Sortable & filterable interactive table
    display_df = df[[
        "author_name", "parsedTime", "caption", "playCount", 
        "diggCount", "commentCount", "shareCount", "collectCount", 
        "webVideoUrl"
    ]].copy()
    
    display_df.rename(columns={
        "author_name": "Username",
        "parsedTime": "Tanggal",
        "caption": "Caption",
        "playCount": "Plays",
        "diggCount": "Likes",
        "commentCount": "Komentar",
        "shareCount": "Shares",
        "collectCount": "Collections",
        "webVideoUrl": "Link Video"
    }, inplace=True)
    
    st.dataframe(
        display_df.sort_values(by="Komentar", ascending=False),
        column_config={
            "Link Video": st.column_config.LinkColumn("Link Video", display_text="Open TikTok ↗"),
            "Tanggal": st.column_config.DatetimeColumn("Tanggal", format="YYYY-MM-DD HH:mm"),
            "Komentar": st.column_config.NumberColumn("Komentar", format="%d")
        },
        use_container_width=True,
        hide_index=True
    )
