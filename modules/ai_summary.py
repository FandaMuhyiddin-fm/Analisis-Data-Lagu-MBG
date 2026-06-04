import streamlit as st
import pandas as pd
import requests
import json
import os
from modules.engagement import calculate_engagement_score

def generate_context_string(videos_df, classified_comments_df):
    """
    Builds a summary context of the data to send to the Claude API.
    Uses the pre-classified sentiments from the provided dataframe.
    """
    # Calculate engagement scores
    v_df = calculate_engagement_score(videos_df.copy())
    top_5_videos = v_df.sort_values(by="engagement_score", ascending=False).head(5)
    
    sentiment_counts = classified_comments_df["sentiment"].value_counts()
    total_comments = len(classified_comments_df)
    pos_count = sentiment_counts.get("Positive", 0)
    neu_count = sentiment_counts.get("Neutral", 0)
    neg_count = sentiment_counts.get("Negative", 0)
    
    pos_pct = (pos_count / total_comments * 100) if total_comments > 0 else 0
    neu_pct = (neu_count / total_comments * 100) if total_comments > 0 else 0
    neg_pct = (neg_count / total_comments * 100) if total_comments > 0 else 0

    # Get sample comments
    pos_samples = classified_comments_df[classified_comments_df["sentiment"] == "Positive"]["comment_text"].head(5).tolist()
    neg_samples = classified_comments_df[classified_comments_df["sentiment"] == "Negative"]["comment_text"].head(5).tolist()

    context = f"""=== METRIK UTAMA ===
Total Video: {len(v_df)}
Total Play (Views): {v_df['playCount'].sum():,}
Total Like: {v_df['diggCount'].sum():,}
Total Komentar: {v_df['commentCount'].sum():,}

=== STATISTIK SENTIMEN KOMENTAR ===
Positif: {pos_count} ({pos_pct:.1f}%)
Netral: {neu_count} ({neu_pct:.1f}%)
Negatif: {neg_count} ({neg_pct:.1f}%)

=== TOP 5 VIDEO TERBAIK (ENGAGEMENT SCORE) ===
"""
    for idx, (_, row) in enumerate(top_5_videos.iterrows(), 1):
        context += f"{idx}. @{row['author_name']} - Caption: {row['caption'][:80]}...\n"
        context += f"   (Plays: {row['playCount']:,} | Likes: {row['diggCount']:,} | Comments: {row['commentCount']:,} | Score: {row['engagement_score']:.1f})\n"

    context += "\n=== CONTOH KOMENTAR POSITIF ===\n"
    for sample in pos_samples:
        context += f"- {sample}\n"

    context += "\n=== CONTOH KOMENTAR NEGATIF ===\n"
    for sample in neg_samples:
        context += f"- {sample}\n"

    return context

def call_claude_api(api_key, context_str):
    """
    Sends the data context to Claude API using the official SDK or requests fallback.
    Instructs the AI to summarize main findings in 3-5 bullet points first (WAJIB 5).
    """
    prompt = f"""Anda adalah seorang analis media sosial dan AI Specialist.
Berikut adalah data analytics dan analisis sentimen dari kampanye video TikTok kami mengenai lagu MBG (Mas Bahlil Ganteng):

{context_str}

Berdasarkan data di atas, tolong buatkan laporan analisis naratif yang komprehensif, menarik, dan berwawasan profesional dalam Bahasa Indonesia. 

Laporan HARUS mengikuti kriteria berikut:
1. **Ringkasan Temuan Utama (3-5 Poin)**: Tuliskan 3 hingga 5 poin ringkasan temuan utama paling penting di bagian paling atas laporan secara eksplisit dan singkat.
2. **Ringkasan Eksekutif (Executive Summary)**: Gambaran umum tentang performa kampanye video TikTok ini.
3. **Analisis Driver Engagement**: Apa yang membuat video-video teratas berhasil mendapatkan engagement yang sangat tinggi?
4. **Analisis Sentimen Komunitas**: Apa topik utama yang dibicarakan netizen pada komentar positif dan komentar negatif?
5. **Rekomendasi Strategis**: Langkah konkret apa yang harus dilakukan tim konten pada kampanye video berikutnya?

Buat laporan dengan format markdown yang rapi, profesional, dan mudah dibaca."""

    # Try official Anthropic SDK first
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1500,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e_sdk:
        # Fallback to direct HTTP Request
        try:
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": "claude-haiku-4-5",
                "max_tokens": 1500,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}]
            }
            res = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=30)
            if res.status_code == 200:
                data = res.json()
                return data["content"][0]["text"]
            else:
                return f"Gagal memanggil Claude API. Status Code: {res.status_code}\nDetail Error: {res.text}"
        except Exception as e_req:
            return f"Error saat menghubungi API: {e_req} (SDK Error: {e_sdk})"

def render_ai_summary_tab(videos_df, classified_comments_df):
    """
    Renders Tab 4: AI Summary and reporting.
    """
    st.markdown("### AI Summary (Powered by Claude)")
    st.markdown("Menghasilkan ringkasan laporan analisis naratif secara otomatis berdasarkan seluruh performa video dan sentimen komentar.")

    # Get API key from dotenv, st.secrets, or manual input (prioritize sidebar input if configured in app.py)
    # Check if effective key exists in session state or app.py's sidebar text input
    effective_key = os.getenv("ANTHROPIC_API_KEY", "")
    
    # We will look up if sidebar key has been set in session state or streamlit memory
    # Try to grab the input key via sidebar access if it was rendered
    # In app.py we rendered st.sidebar.text_input with value. Streamlit stores widgets in Session State if a key is provided,
    # but we can also just fetch it if the user configured it.
    
    # Generate context
    context_str = generate_context_string(videos_df, classified_comments_df)

    # # Context preview
    # with st.expander("Lihat Data Context yang Dikirim ke Claude"):
    #     st.code(context_str, language="text")

    # Enter API key manually if not found in .env
    # manual_key = st.text_input("Konfirmasi Anthropic API Key (jika belum diisi di sidebar/.env):", value="", type="password")
    
    # final_key = manual_key if manual_key else effective_key
    final_key = effective_key

    # Action button
    if st.button("Generate Laporan Analisis AI"):
        if not final_key.strip():
            st.error("Gagal: API Key tidak ditemukan! Harap isi API Key Anda di sidebar atau input di atas.")
        else:
            with st.spinner("Claude sedang menganalisa data Anda..."):
                report = call_claude_api(final_key, context_str)
                st.session_state["ai_report"] = report

    # Display report if available
    if "ai_report" in st.session_state:
        st.markdown("---")
        st.markdown("### Hasil Laporan Analisis AI")
        st.markdown(st.session_state["ai_report"])

        # Export report
        st.download_button(
            label="Ekspor Laporan (.txt)",
            data=st.session_state["ai_report"],
            file_name="Laporan_Analisis_Sentimen_TikTok_Claude.txt",
            mime="text/plain"
        )
