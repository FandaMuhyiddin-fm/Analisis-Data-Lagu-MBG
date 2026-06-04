import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import re

# Fallback Rule-Based Sentiment Analysis
POSITIVE_WORDS = {
    "lucu", "ngakak", "wkwk", "hahaha", "candu", "nagih", "terngiang", "seru", 
    "hiburan", "kreatif", "gokil", "kocak", "receh", "menghibur", "gemoy", "kece",
    "bagus", "mantap", "keren", "top", "viral", "suka", "glowing", "cakep"
}

NEGATIVE_WORDS = {
    "cringe", "geli", "aneh", "ilfil", "norak", "maksa", "pencitraan", "pesanan",
    "gajelas", "ga jelas", "jelek", "buruk", "bosan", "enek", "eneg", "Geleuh",
    "sarkas", "sindiran", "hujat", "plagiat", "gagal", "kecewa", "parah", "malu",
    "Settingan", "gimmick", "gimik"
}

def analyze_sentiment_rules(text):
    """
    Simple rule-based sentiment classifier for Indonesian/English text fallback.
    """
    if not isinstance(text, str):
        return "Neutral"
    
    text = text.lower()
    pos_count = sum(1 for word in POSITIVE_WORDS if word in text)
    neg_count = sum(1 for word in NEGATIVE_WORDS if word in text)
    
    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment_vader(text, analyzer):
    """
    Analyze sentiment using VADER.
    """
    if not isinstance(text, str):
        return "Neutral"
    
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

@st.cache_resource
def load_indobert_pipeline():
    """
    Loads IndoBERT pipeline. Caches resource to prevent reloading.
    """
    try:
        from transformers import pipeline
        import os
        token = os.getenv("HF_TOKEN", None)
        # mdhugol/indonesia-bert-sentiment-classification
        # labels: 0 -> positive, 1 -> neutral, 2 -> negative
        pipe = pipeline(
            "text-classification", 
            model="mdhugol/indonesia-bert-sentiment-classification",
            tokenizer="mdhugol/indonesia-bert-sentiment-classification",
            token=token
        )
        return pipe
    except Exception as e:
        # IndoBERT loading failed (e.g. no transformers/torch or connection issue)
        return None

@st.cache_data
def classify_comments(comments_df, model_choice="VADER"):
    """
    Classify all comments based on selected model (IndoBERT, VADER, Rule-Based).
    Caches the output dataframe to prevent lag.
    """
    if comments_df.empty:
        return comments_df
        
    df = comments_df.copy()
    
    # Check if we already have precomputed sentiment column in the dataset (instant load)
    if "sentiment" in df.columns:
        return df
        
    if model_choice == "IndoBERT":
        if "sentiment_indobert" in df.columns:
            df["sentiment"] = df["sentiment_indobert"]
            return df
        nlp = load_indobert_pipeline()
        if nlp is not None:
            sentiments = []
            # Predict in batches to prevent UI freeze
            texts = df["comment_text"].fillna("").tolist()
            # Clean texts briefly for model safety
            texts = [t[:512] for t in texts] # limit chars
            try:
                # Run pipeline
                results = nlp(texts)
                for res in results:
                    label = res["label"]
                    # Map IndoBERT labels: mdhugol maps label 0 = positive, 1 = neutral, 2 = negative
                    if label == "LABEL_0" or label == "0" or label == "positive":
                        sentiments.append("Positive")
                    elif label == "LABEL_1" or label == "1" or label == "neutral":
                        sentiments.append("Neutral")
                    else:
                        sentiments.append("Negative")
                df["sentiment"] = sentiments
                return df
            except Exception as e:
                model_choice = "Rule-Based"
        else:
            model_choice = "Rule-Based"

    if model_choice == "VADER":
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            df["sentiment"] = df["comment_text"].apply(lambda x: analyze_sentiment_vader(x, analyzer))
            return df
        except Exception as e:
            model_choice = "Rule-Based"

    # Rule-Based fallback (extremely fast: runs in <0.05 seconds for 7000 rows)
    df["sentiment"] = df["comment_text"].apply(analyze_sentiment_rules)
    return df

def generate_wordcloud(text):
    """
    Generates a word cloud image and returns it as a bytes buffer.
    """
    if not text.strip():
        # Fallback empty image
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Tidak ada kata untuk dianalisa", ha="center", va="center", fontsize=14, color="white")
        ax.axis("off")
        fig.patch.set_facecolor("#0e1117")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
        plt.close(fig)
        return buf.getvalue()

    try:
        # Clean up common stop words
        stopwords = {
            "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu", "ada",
            "bisa", "aja", "bgt", "saja", "gak", "ga", "gua", "gue", "gw", "aku", "kamu", "dia",
            "kita", "mereka", "lagi", "udah", "sudah", "buat", "tapi", "akan", "tahu", "tonton"
        }
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color="#0e1117",
            colormap="rainbow", 
            stopwords=stopwords,
            collocations=False
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        fig.patch.set_facecolor("#0e1117")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        st.error(f"Error generating WordCloud: {e}")
        return None

def render_sentiment_tab(videos_df, comments_df):
    """
    Renders Tab 2: Sentiment analysis details.
    """
    if comments_df.empty:
        st.warning("Tidak ada data komentar untuk dianalisa.")
        return

    # Reuse classified comments directly or run default
    with st.spinner("Menganalisa sentimen komentar..."):
        classified_comments = classify_comments(comments_df, "IndoBERT")

    # Video selector
    st.markdown("### Analisis Sentimen per Video")
    
    # Generate choice list
    video_choices = ["Semua Video (Gabungan)"]
    video_map = {}
    
    # Filter videos that actually have comments
    videos_with_comments = videos_df[videos_df["videoId"].isin(classified_comments["videoId"])].copy()
    if videos_with_comments.empty:
        videos_with_comments = videos_df
        
    for _, row in videos_with_comments.iterrows():
        cap = row["caption"] if pd.notna(row["caption"]) and row["caption"] else "[No Caption]"
        display_label = f"@{row['author_name']} - {cap[:60]}... ({row['videoId']})"
        video_choices.append(display_label)
        video_map[display_label] = row["videoId"]

    selected_video_label = st.selectbox("Pilih Video untuk Dianalisa:", video_choices)

    # Filter comments based on selection
    if selected_video_label == "Semua Video (Gabungan)":
        filtered_comments = classified_comments
        selected_video_row = None
    else:
        video_id = video_map[selected_video_label]
        filtered_comments = classified_comments[classified_comments["videoId"] == video_id]
        selected_video_row = videos_df[videos_df["videoId"] == video_id].iloc[0] if not videos_df[videos_df["videoId"] == video_id].empty else None

    # Calculate statistics
    total_comments = len(filtered_comments)
    if total_comments == 0:
        st.warning("Tidak ada komentar ditemukan untuk video ini.")
        return

    sentiment_counts = filtered_comments["sentiment"].value_counts()
    pos_count = sentiment_counts.get("Positive", 0)
    neu_count = sentiment_counts.get("Neutral", 0)
    neg_count = sentiment_counts.get("Negative", 0)

    pos_pct = (pos_count / total_comments) * 100
    neu_pct = (neu_count / total_comments) * 100
    neg_pct = (neg_count / total_comments) * 100

    # UI columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Distribusi Sentimen")
        # Donut chart
        chart_data = pd.DataFrame({
            "Sentimen": ["Positif", "Netral", "Negatif"],
            "Jumlah": [pos_count, neu_count, neg_count],
            "Warna": ["#2ecc71", "#95a5a6", "#e74c3c"]
        })
        
        fig = px.pie(
            chart_data, 
            values="Jumlah", 
            names="Sentimen", 
            hole=0.5,
            color="Sentimen",
            color_discrete_map={
                "Positif": "#2ecc71",
                "Netral": "#7f8c8d",
                "Negatif": "#e74c3c"
            },
            template="plotly_dark"
        )
        fig.update_layout(showlegend=True, height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Metrics display
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #2ecc71; font-weight: bold;">🟢 Positif:</span>
                    <span><b>{pos_count}</b> ({pos_pct:.1f}%)</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #f1c40f; font-weight: bold;">🟡 Netral:</span>
                    <span><b>{neu_count}</b> ({neu_pct:.1f}%)</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #e74c3c; font-weight: bold;">🔴 Negatif:</span>
                    <span><b>{neg_count}</b> ({neg_pct:.1f}%)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("#### Word Cloud Komentar")
        all_comments_text = " ".join(filtered_comments["comment_text"].astype(str).tolist())
        
        # Clean text basic regex (remove emojis, stickers, links)
        all_comments_text = re.sub(r'\[Sticker\]', '', all_comments_text)
        all_comments_text = re.sub(r'http\S+', '', all_comments_text)
        all_comments_text = re.sub(r'[^a-zA-Z\s]', '', all_comments_text)
        
        wc_image_bytes = generate_wordcloud(all_comments_text)
        if wc_image_bytes:
            st.image(wc_image_bytes, use_container_width=True)

    st.markdown("---")
    st.markdown("### Contoh Komentar per Kategori (Sentimen)")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    with col_ex1:
        st.success("🟢 Contoh Positif")
        pos_examples = filtered_comments[filtered_comments["sentiment"] == "Positive"]["comment_text"].head(3).tolist()
        if pos_examples:
            for idx, ex in enumerate(pos_examples, 1):
                st.markdown(f"**{idx}.** *\"{ex}\"*")
        else:
            st.info("Tidak ada komentar positif.")
            
    with col_ex2:
        st.warning("🟡 Contoh Netral")
        neu_examples = filtered_comments[filtered_comments["sentiment"] == "Neutral"]["comment_text"].head(3).tolist()
        if neu_examples:
            for idx, ex in enumerate(neu_examples, 1):
                st.markdown(f"**{idx}.** *\"{ex}\"*")
        else:
            st.info("Tidak ada komentar netral.")
            
    with col_ex3:
        st.error("🔴 Contoh Negatif")
        neg_examples = filtered_comments[filtered_comments["sentiment"] == "Negative"]["comment_text"].head(3).tolist()
        if neg_examples:
            for idx, ex in enumerate(neg_examples, 1):
                st.markdown(f"**{idx}.** *\"{ex}\"*")
        else:
            st.info("Tidak ada komentar negatif.")

    st.markdown("---")

    # Comments table
    st.markdown("### Daftar Komentar")
    
    # Filter by sentiment type
    sentiment_filter = st.multiselect(
        "Filter Label Sentimen:", 
        options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"]
    )
    
    table_df = filtered_comments[filtered_comments["sentiment"].isin(sentiment_filter)].copy()
    
    # Display table columns
    display_cols = ["username", "parsedTime", "comment_text", "likeCount", "sentiment"]
    table_df = table_df[display_cols].copy()
    table_df.rename(columns={
        "username": "Username",
        "parsedTime": "Waktu",
        "comment_text": "Komentar",
        "likeCount": "Likes",
        "sentiment": "Sentimen"
    }, inplace=True)

    st.dataframe(
        table_df.sort_values(by="Likes", ascending=False),
        column_config={
            "Waktu": st.column_config.DatetimeColumn("Waktu", format="YYYY-MM-DD HH:mm"),
            "Likes": st.column_config.NumberColumn("Likes", format="%d")
        },
        use_container_width=True,
        hide_index=True
    )
