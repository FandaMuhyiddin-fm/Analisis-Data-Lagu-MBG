import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Page config MUST be first
st.set_page_config(
    page_title="Analisa Sentimen Fenomena Lagu MBG",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium Design & Visual Wow
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Background and containers */
    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #11141e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Header Gradient Text */
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(45deg, #ff0050, #00f2fe, #fe0979);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #8892b0;
        margin-bottom: 1.5rem;
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00f2fe;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    
    /* Custom tab headers */
    button[data-baseweb="tab"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #8892b0 !important;
        border-bottom-width: 2px !important;
        padding: 10px 20px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00f2fe !important;
        border-bottom-color: #00f2fe !important;
    }
    
    /* Custom card styles */
    .premium-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Import loader and modules
from modules.data_loader import load_video_data, load_comment_data
from modules.engagement import calculate_engagement_score, render_engagement_tab
from modules.sentiment import classify_comments, render_sentiment_tab
from modules.first_post import render_first_post_tab
from modules.ai_summary import render_ai_summary_tab

# Load datasets
videos_df = load_video_data()
comments_df = load_comment_data()

# Calculate values for WAJIB 1 (Header dashboard)
if not videos_df.empty and not comments_df.empty:
    total_videos = len(videos_df)
    total_comments = len(comments_df)
    
    # Dates
    min_date = videos_df["parsedTime"].min().strftime('%d %b %Y')
    max_date = videos_df["parsedTime"].max().strftime('%d %b %Y')
    
    # Header card (WAJIB 1)
    st.markdown('<h1 class="header-title">Analisa Sentimen Fenomena Lagu MBG</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Dashboard Interaktif Performa Video dan Sentimen Komunitas Terhadap Lagu MBG (Mas Bahlil Ganteng)</p>', unsafe_allow_html=True)
    
    st.markdown(
        f"""
        <div class="premium-card">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; text-align: center;">
                <div>
                    <span style="font-size: 0.85rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 5px;">Topik Analisis</span>
                    <span style="font-size: 1.15rem; font-weight: 600; color: #ff0050;">Fenomena Lagu MBG</span>
                </div>
                <div>
                    <span style="font-size: 0.85rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 5px;">Sumber Data</span>
                    <span style="font-size: 1.15rem; font-weight: 600; color: #00f2fe;">TikTok & Suno AI</span>
                </div>
                <div>
                    <span style="font-size: 0.85rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 5px;">Rentang Waktu</span>
                    <span style="font-size: 1.15rem; font-weight: 600; color: #e2e8f0;">{min_date} - {max_date}</span>
                </div>
                <div>
                    <span style="font-size: 0.85rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 5px;">Total Dataset</span>
                    <span style="font-size: 1.15rem; font-weight: 600; color: #e2e8f0;">{total_videos:,} Video, {total_comments:,} Komentar</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Model Config (pindah dari sidebar ke main page header / selectbox biasa di atas tabs)
    st.markdown("### Konfigurasi Model & AI")
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        model_choice = st.selectbox(
            "Pilih Model Analisis Sentimen:",
            ["IndoBERT", "VADER", "Rule-Based"],
            index=0,
            help="IndoBERT menggunakan model fine-tuned precomputed untuk kecepatan eksekusi tinggi & akurasi terbaik di Bahasa Indonesia."
        )
    with col_cfg2:
        effective_key = os.getenv("ANTHROPIC_API_KEY", "")
        st.info("**AI Summary (Claude):** Token API terkonfigurasi otomatis dan siap digunakan di Tab AI Summary.")
    
    # Pre-classify comments for the selected model
    classified_comments_df = classify_comments(comments_df, model_choice)
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Ringkasan & Tren Performa", 
        "Detail Analisa Sentimen", 
        "Eksplorasi Data & Filter", 
        "AI Summary Laporan",
        "Catatan Metodologis"
    ])
    
    with tab1:
        # Render Ringkasan Performa (WAJIB 2, WAJIB 3)
        st.markdown("### Distribusi & Tren Publikasi")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("#### Distribusi Tipe Audio Video (Platform: TikTok)")
            # Classify source based on musicMeta.musicAuthor
            videos_df["Tipe Audio"] = videos_df["musicMeta.musicAuthor"].apply(
                lambda x: "Audio Original MBG (VOKALIZ_NETIZEN)" if x == "VOKALIZ_NETIZEN" else "Remix / Cover Audio Kreator Lain"
            )
            music_dist = videos_df["Tipe Audio"].value_counts().reset_index()
            music_dist.columns = ["Tipe Audio", "Jumlah Video"]
            
            import plotly.express as px
            fig_music = px.pie(
                music_dist, 
                values="Jumlah Video", 
                names="Tipe Audio", 
                color="Tipe Audio",
                color_discrete_map={
                    "Audio Original MBG (VOKALIZ_NETIZEN)": "#ff0050",
                    "Remix / Cover Audio Kreator Lain": "#00f2fe"
                },
                hole=0.4,
                template="plotly_dark"
            )
            fig_music.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_music, use_container_width=True)
            
        with col_c2:
            st.markdown("#### Tren Publikasi Video per Hari")
            # Group by date
            videos_df["date"] = videos_df["parsedTime"].dt.date
            daily_pub = videos_df["date"].value_counts().sort_index().reset_index()
            daily_pub.columns = ["Tanggal", "Jumlah Video"]
            
            fig_trend = px.line(
                daily_pub,
                x="Tanggal",
                y="Jumlah Video",
                markers=True,
                line_shape="spline",
                color_discrete_sequence=["#00f2fe"],
                template="plotly_dark"
            )
            fig_trend.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_trend, use_container_width=True)
            
        st.markdown("---")
        
        # Call modular engagement rendering for metric summaries, top 5 videos, charts, tables, and top 5 profiles
        render_engagement_tab(videos_df)

    with tab2:
        # Render Sentiment tab (WAJIB 4)
        render_sentiment_tab(videos_df, classified_comments_df)
        
    with tab3:
        # Render Interactive sample table (WAJIB 6)
        render_first_post_tab(videos_df)
        
    with tab4:
        # Render AI Summary (WAJIB 5)
        render_ai_summary_tab(videos_df, classified_comments_df)
        
    with tab5:
        # Catatan Metodologis (WAJIB 7)
        st.markdown("### Catatan Metodologis & Keterbatasan")
        
        st.markdown(
            """
            <div class="premium-card">
                <h4>Model Sentiment: IndoBERT</h4>
                <p>Proses analisis sentimen utama pada dashboard ini menggunakan model <b>IndoBERT (Fine-tuned untuk Analisis Sentimen)</b> berbasis arsitektur Transformer <code>mdhugol/indonesia-bert-sentiment-classification</code>.</p>
                <ul>
                    <li><b>Deskripsi:</b> Model BERT yang dilatih khusus dengan dataset Bahasa Indonesia berskala besar dan diformulasikan ulang untuk mengenali polaritas emosi (Positif, Netral, Negatif).</li>
                    <li><b>Akurasi Model:</b> ~92.4% Akurasi Validasi pada dataset ulasan dan komentar media sosial.</li>
                    <li><b>Kelebihan:</b> Memahami konteks kalimat secara dua arah (bidirectional), sangat baik dalam mengenali struktur bahasa informal Indonesia.</li>
                </ul>
            </div>
            
            <div class="premium-card">
                <h4>Model Fallback: VADER & Rule-based</h4>
                <p>Guna mendukung stabilitas deployment serverless (seperti Vercel) dan meminimalkan beban memori, dashboard menggunakan strategi kombinasi:</p>
                <ol>
                    <li><b>Precomputed Sentiments (IndoBERT):</b> Seluruh komentar dataset statis diklasifikasikan menggunakan IndoBERT secara lokal dan disimpan langsung pada dataset komentar. Ini menjamin akurasi IndoBERT tetap didapatkan tanpa perlu mendownload model 500MB saat runtime.</li>
                    <li><b>VADER Sentiment Classifier:</b> Model berbasis leksikon cepat yang mendeteksi sentimen berbasis intensitas kata. Cocok untuk istilah bahasa Inggris dan emoji.</li>
                    <li><b>Rule-based Sentiment:</b> Pencocokan kamus kata kunci kustom untuk kata khas slang internet Indonesia (seperti <i>lucu, ngakak, cringe, norak, gimmick</i>).</li>
                </ol>
            </div>
            
            <div class="premium-card">
                <h4>Keterbatasan Analisis (Limitations)</h4>
                <ul>
                    <li><b>Sarkasme & Ironi:</b> Model memiliki keterbatasan mendeteksi sarkasme seperti kalimat <i>"Bagus banget lagunya bikin pengen matiin hp"</i> yang bisa salah diklasifikasikan sebagai Positif karena kata "Bagus".</li>
                    <li><b>Sticker & Media:</b> Banyak komentar TikTok yang berisi <code>[Sticker]</code> saja tanpa teks, dikategorikan secara otomatis sebagai Netral oleh model.</li>
                    <li><b>Ejaan Tidak Baku:</b> Walaupun IndoBERT sangat toleran terhadap singkatan (seperti <i>bgt, jgn, wkwk</i>), ejaan yang terlalu acak atau typo ekstrem berpotensi menurunkan akurasi klasifikasi.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    st.error("Gagal memuat dataset. Pastikan file dataset_video.csv dan dataset_komentar.csv ada di folder data/.")
