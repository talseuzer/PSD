# Import library-library utama yang dibutuhkan untuk aplikasi
import streamlit as st
import pandas as pd
import re
import torch
import transformers
from transformers import pipeline
from io import BytesIO
import logging
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# DAFTAR AKUN RESMI (ALLOWLIST)
AKUN_RESMI_UNIKOM = [
    'pmbftik', 'feb_unikom', 'fisipunikom_offic', 'pascasarjana_unikom', 'pwkunikom',
    'accounting.unikom', 'prodi.manajemen', 'akuntansidiploma3unikom', 'mpd3_unikom',
    'kpd3_unikom', 'ilmupemerintahan86', 'ik.unikom', 'dkv.unikom', 'hiunikomofficial',
    'msi.unikom', 'utv.id', 'hitsunikomradio', 'dau.unikom', 'penerimaan_mahasiswabaru',
    'careercenterunikom', 'unikomcodelabs', 'pusba.unikom'
]

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1: FUNGSI-FUNGSI PEMROSESAN DATA (EXISTING + UPGRADE)
def muat_data(uploaded_file):
    """Memuat data dari file Excel atau CSV."""
    try:
        if uploaded_file.name.endswith('.csv'):
            file_content = uploaded_file.getvalue().decode('utf-8')
            lines = file_content.strip().split('\n')
            if lines and 'text,"diggCount"' in lines[0]: lines = lines[1:]
            processed_lines = [re.split(r',(?=")', line) for line in lines]
            df = pd.DataFrame(processed_lines)
            
            column_names = ['text', 'diggCount', 'replyCommentTotal', 'createTimeISO', 'uniqueId', 'videoWebUrl', 'uid', 'cid', 'avatarThumbnail']
            num_columns_to_name = min(len(df.columns), len(column_names))
            df = df.iloc[:, :num_columns_to_name]
            df.columns = column_names[:num_columns_to_name]
            
            for col in df.columns:
                if df[col].dtype == 'object': df[col] = df[col].str.strip('"')
            return df
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal memuat file: {e}")
        return None

def deteksi_kategori(nama_akun):
    """
    Mendeteksi kategori unit. 
    Prioritas 1: Cek daftar spesifik (Hardcoded).
    Prioritas 2: Tebak berdasarkan kata kunci (Fallback).
    """
    nama = str(nama_akun).lower().strip()
    
    # 1. Daftar Mapping Spesifik (TikTok UNIKOM)
    mapping_spesifik = {
        # Media
        'utv.id': 'Media Kampus',
        'hitsunikomradio': 'Media Kampus',
        
        # Biro & Layanan Pusat
        'penerimaan_mahasiswabaru': 'Birokrasi/Layanan',
        'careercenterunikom': 'Birokrasi/Layanan',
        'dau.unikom': 'Birokrasi/Layanan',
        'pusba.unikom': 'Birokrasi/Layanan',
        
        # Komunitas / Riset
        'unikomcodelabs': 'Komunitas/Riset',
        
        # Akademik (Fakultas & Prodi) - Default catch-all
        'feb_unikom': 'Akademik (Fakultas/Prodi)',
        'fisipunikom_offic': 'Akademik (Fakultas/Prodi)',
        'pascasarjana_unikom': 'Akademik (Fakultas/Prodi)',
        'pmbftik': 'Akademik (Fakultas/Prodi)',
        'pwkunikom': 'Akademik (Fakultas/Prodi)',
        'accounting.unikom': 'Akademik (Fakultas/Prodi)',
        'prodi.manajemen': 'Akademik (Fakultas/Prodi)',
        'akuntansidiploma3unikom': 'Akademik (Fakultas/Prodi)',
        'mpd3_unikom': 'Akademik (Fakultas/Prodi)',
        'kpd3_unikom': 'Akademik (Fakultas/Prodi)',
        'ilmupemerintahan86': 'Akademik (Fakultas/Prodi)',
        'ik.unikom': 'Akademik (Fakultas/Prodi)',
        'dkv.unikom': 'Akademik (Fakultas/Prodi)',
        'hiunikomofficial': 'Akademik (Fakultas/Prodi)',
        'msi.unikom': 'Akademik (Fakultas/Prodi)'
    }
    
    # Cek apakah nama akun ada di daftar spesifik
    if nama in mapping_spesifik:
        return mapping_spesifik[nama]
    
    # 2. Logika Cadangan (Smart Guessing) - Untuk Akun Instagram nanti
    if any(x in nama for x in ['tv', 'radio', 'news', 'media']):
        return 'Media Kampus'
    elif any(x in nama for x in ['bem', 'hima', 'ukm', 'unit', 'pers', 'kema', 'codelabs']):
        return 'Organisasi Mahasiswa (UKM)'
    elif any(x in nama for x in ['pmb', 'admisi', 'humas', 'biro', 'sekretariat', 'career', 'pusba']):
        return 'Birokrasi/Layanan'
    elif any(x in nama for x in ['fakultas', 'prodi', 'jurusan', 'teknik', 'ilmu', 'sastra', 'ekonomi', 'hukum', 'fisip', 'magister', 'diploma']):
        return 'Akademik (Fakultas/Prodi)'
    else:
        return 'Lainnya/Umum'

def proses_dan_analisis(df: pd.DataFrame, map_kolom: dict, model_name: str):
    # Rename kolom sesuai mapping
    df.rename(columns=map_kolom, inplace=True)
    
    df['teks_asli'] = df['teks_komentar'].astype(str)
    
    # Ekstraksi Username dari URL
    def bersihkan_nama_akun(teks):
        teks = str(teks)
        match_tiktok = re.search(r'@([a-zA-Z0-9_\.]+)', teks)
        if match_tiktok: return match_tiktok.group(1)
        match_ig = re.search(r'instagram\.com/([a-zA-Z0-9_\.]+)', teks)
        if match_ig: return match_ig.group(1)
        if 'http' not in teks: return re.sub(r'[^a-zA-Z0-9_\.]', '', teks) 
        return teks

    if 'sumber_akun' in df.columns:
        df.dropna(subset=['sumber_akun'], inplace=True)
        df['sumber_akun'] = df['sumber_akun'].apply(bersihkan_nama_akun)
        df['kategori_unit'] = df['sumber_akun'].apply(deteksi_kategori)
    else:
        df['sumber_akun'] = "Unknown"
        df['kategori_unit'] = "Uncategorized"

    # Cleaning Basic Tanggal
    if 'tanggal' in df.columns:
        df['tanggal'] = pd.to_datetime(df['tanggal'], format='mixed', utc=True, errors='coerce').dt.date
    else:
        df['tanggal'] = "N/A"

    # Load Kamus Slang
    @st.cache_data
    def get_kamus_slang():
        try:
            url = 'https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv'
            df_slang = pd.read_csv(url)
            return dict(zip(df_slang['slang'], df_slang['formal']))
        except: return {}

    kamus_slang = get_kamus_slang()
    kamus_slang.update({'yg': 'yang', 'tdk': 'tidak', 'gak': 'tidak', 'bgt': 'banget', 'anjir': 'hebat',
                        'gueh': 'saya', 'gue': 'saya', 'gua': 'saya', 'aku': 'saya', 'akuh': 'saya', 
                        'enggak': 'tidak', 'bgtt': 'banget', 'letsgoo': 'ayo', 'nyoblos': 'coblos', '': '', 
                        '': '', '': '', '': '', '': '', '': '', }) 

    def clean_regex(t):
        t = str(t).lower()
        t = re.sub(r'https?://\S+|www\.\S+', '', t)
        t = re.sub(r'@[a-zA-Z0-9_]+', '', t)
        t = re.sub(r'#\w+', '', t)
        return t.strip()
    df['teks_clean_regex'] = df['teks_asli'].apply(clean_regex)

    def fix_slang(t):
        kata_kata = str(t).split()
        return " ".join([kamus_slang.get(k, k) for k in kata_kata])
    df['teks_slang_fixed'] = df['teks_clean_regex'].apply(fix_slang)

    df['teks_final'] = df['teks_slang_fixed'].apply(lambda x: re.sub(r'(.)\1{2,}', r'\1\1', str(x)))

    # 6. Sentiment Analysis (IndoBERT)
    @st.cache_resource
    def get_analyzer(model_name):
        device = 0 if torch.cuda.is_available() else -1
        return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=device)

    sentiment_analyzer = get_analyzer(model_name)
    
    comments = df['teks_final'].tolist()
    
    hasil_sentimen = []
    progress_bar = st.progress(0, text="Menganalisis sentimen dengan IndoBERT...")
    
    for i, teks in enumerate(comments):
        try:
            hasil = sentiment_analyzer(teks, truncation=True, max_length=512)
            hasil_sentimen.append(hasil[0])
        except:
            hasil_sentimen.append({'label': 'neutral', 'score': 0.5}) 
        progress_bar.progress((i + 1) / len(comments))

    df_hasil = pd.DataFrame(hasil_sentimen)
    df['sentimen'] = df_res = df_hasil['label'].values
    df['skor_conf'] = df_hasil['score'].values
    
    return df

# # =========================================================
# # TEMPORARY ANALYTICS BLOCK (VERSION 2 - UI OPTIMIZED)
# # =========================================================
# if st.session_state.get('hasil_analisis') is not None:
#     st.sidebar.markdown("---")
#     temp_menu = st.sidebar.radio("📑 Halaman Sementara:", ["Dashboard Utama", "Tabel TF-IDF per Unit", "Evaluasi Model"])

#     df_temp = st.session_state.hasil_analisis

#     if temp_menu == "Tabel TF-IDF per Unit":
#         st.header("📊 Tabel Bobot Kata TF-IDF per Unit")
#         from sklearn.feature_extraction.text import TfidfVectorizer
        
#         units = [u for u in df_temp['kategori_unit'].unique() if u != 'Uncategorized']
        
#         # Menggunakan kolom agar tabel tidak melebar memenuhi layar
#         cols = st.columns(3) 
#         for i, unit in enumerate(units):
#             unit_data = df_temp[df_temp['kategori_unit'] == unit]['teks_final']
#             if len(unit_data) > 2:
#                 vectorizer = TfidfVectorizer(max_features=10)
#                 tfidf_matrix = vectorizer.fit_transform(unit_data)
#                 importance = tfidf_matrix.toarray().mean(axis=0)
#                 words = vectorizer.get_feature_names_out()
                
#                 df_tfidf = pd.DataFrame({'Kata': words, 'Skor': importance}).sort_values(by='Skor', ascending=False)
                
#                 # Masukkan tabel ke kolom secara bergantian (0, 1, 2)
#                 with cols[i % 3]:
#                     st.subheader(f"📍 {unit}")
#                     st.dataframe(df_tfidf, hide_index=True, use_container_width=True)
#         st.stop()

#     elif temp_menu == "Evaluasi Model":
#         st.header("📈 Evaluasi Keyakinan Model")
#         import matplotlib.pyplot as plt

#         # Statistik Ringkas
#         c1, c2 = st.columns(2)
#         with c1: st.metric("Rata-rata Confidence", f"{df_temp['skor_conf'].mean():.2%}")
#         with c2: st.metric("Data Total", len(df_temp))

#         # Grafik Distribusi dengan Matplotlib (Lebih fleksibel untuk Sumbu X)
#         st.subheader("Distribusi Skor Keyakinan")
#         fig, ax = plt.subplots(figsize=(10, 4))
        
#         # Membuat histogram dengan 10 bin agar batang lebih rapat
#         n, bins, patches = ax.hist(df_temp['skor_conf'], bins=10, color='#1f77b4', edgecolor='white', rwidth=0.9)
        
#         ax.set_xlabel('Rentang Skor Confidence (0.0 - 1.0)')
#         ax.set_ylabel('Jumlah Komentar')
#         plt.xticks(bins, rotation=45) # Rotasi label sumbu X agar terbaca
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
        
#         st.pyplot(fig)

        

#         st.subheader("⚠️ Audit Data Skor Rendah (< 0.6)")
#         # Tabel Audit Trail dibuat lebih ramping
#         st.dataframe(
#             df_temp[df_temp['skor_conf'] < 0.6][['teks_asli', 'sentimen', 'skor_conf']],
#             column_config={
#                 "teks_asli": st.column_config.TextColumn("Komentar", width="large"),
#                 "skor_conf": st.column_config.NumberColumn("Conf", format="%.2f")
#             },
#             hide_index=True,
#             use_container_width=True
#         )
#         st.stop()
# # =========================================================

# 2: UI APLIKASI (UPDATED)

st.set_page_config(layout="wide", page_title="UNIKOM Sentiment Wisdom Tool")
st.title("Analisis Sentiment & Wawasan Reputasi Sosial Media UNIKOM")
st.markdown("Bertujuan untuk mendapatkan Wawasan dari komentar media sosial pada akun UNIKOM.")

if 'hasil_analisis' not in st.session_state: st.session_state.hasil_analisis = None
if st.session_state.hasil_analisis is not None:
    st.subheader("Data Post-Processing")
    
    # Memilih kolom spesifik untuk efisiensi tampilan
    kolom_audit = [
        'sumber_akun', 
        'teks_asli', 
        'teks_clean_regex', 
        'teks_slang_fixed', 
        'teks_final', 
        'sentimen', 
        'skor_conf'
    ]
    
    # Menampilkan tabel dengan expander agar tidak memakan ruang jika data banyak
    with st.expander("Detail data setelah Preprocessing dan Pelabelan"):
        st.dataframe(
            st.session_state.hasil_analisis[kolom_audit], 
            use_container_width=True,
            column_config={
                "skor_conf": st.column_config.NumberColumn("Confidence", format="%.2f")
            }
        )

with st.sidebar:
    st.header("1. Input Data")
    uploaded_file = st.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])

    # Placeholder container agar opsi muncul setelah data di-load
    container_akun = st.container()

# --- PROSES DATA ---
if uploaded_file is not None:
    if 'nama_file' not in st.session_state or st.session_state.nama_file != uploaded_file.name:
        st.session_state.hasil_analisis = None
        st.session_state.nama_file = uploaded_file.name
    
    df_input = muat_data(uploaded_file)
    
    if df_input is not None and st.session_state.hasil_analisis is None:
        # [RESTORED] Pratinjau Data Mentah
        st.subheader("1. Pratinjau Data Mentah")
        st.markdown("Pastikan data terbaca dengan benar sebelum mapping.")
        st.dataframe(df_input.head())
        st.markdown("---")

        st.subheader("2. Mapping Kolom")
        st.info("Pilih 'Tidak Ada' jika kolom tersebut tidak tersedia di file Anda.")
        
        cols = df_input.columns.tolist()
        opsi_kolom = ["-- Tidak Ada --"] + cols
        
        col1, col2, col3 = st.columns(3)
        
        # 1. Komentar (Wajib)
        map_komentar = col1.selectbox("Kolom Komentar (Wajib)", cols, index=0)
        
        # 2. Sumber Akun (Smart Detection)
        idx_sumber = 0 
        for i, col in enumerate(cols):
            if any(x in col.lower() for x in ['user', 'unique', 'sumber', 'account']):
                idx_sumber = i + 1 
                break
        map_sumber = col2.selectbox("Kolom Sumber Akun", opsi_kolom, index=idx_sumber)
        
        # 3. Tanggal (Smart Detection)
        idx_tanggal = 0
        for i, col in enumerate(cols):
            if any(x in col.lower() for x in ['time', 'date', 'tanggal', 'waktu']):
                idx_tanggal = i + 1 
                break
        map_tanggal = col3.selectbox("Kolom Tanggal", opsi_kolom, index=idx_tanggal)
        
        if map_sumber == "-- Tidak Ada --":
            st.warning("Tanpa 'Kolom Nama Akun', fitur Perbandingan Unit & Magic Quadrant tidak akan berfungsi.")
        
        if st.button("Mulai Analisis Data"):
            mapping = {map_komentar: 'teks_komentar'}
            if map_sumber != "-- Tidak Ada --": mapping[map_sumber] = 'sumber_akun'
            if map_tanggal != "-- Tidak Ada --": mapping[map_tanggal] = 'tanggal'
            
            st.session_state.hasil_analisis = proses_dan_analisis(df_input.copy(), mapping, "w11wo/indonesian-roberta-base-sentiment-classifier")
            st.rerun()

def generate_wc(data, color):
    # Hardcoded Stopwords
    custom_stopwords = set([
        'dan', 'yang', 'di', 'ke', 'dari', 'itu', 'ini', 'untuk', 'dengan', 
        'adalah', 'ya', 'gak', 'sudah', 'aja', 'ada', 'bisa', 'kalau', 'ga', 
        'banget', 'bgt', 'si', 'nih', 'kalo', 'sama', 'buat', 'juga', 'unikom',
        'kak', 'bang', 'admin', 'min', 'terima', 'kasih', 'deh', 'nya', 'banyak',
        'aku', 'kan', 'kok', 'tuh', 'ku', 'terus', 'nab', 'saja', 'saya', 'bun',
        'nurullss', 'windii', 'bu', 'r', 'bukan', 'wkwk', 'gong',
        'ey', 'seru', 'bunda', 'kali', 'ouu', 'kita'
    ])
    
    # Gabungkan teks
    text = " ".join(data.dropna().astype(str).tolist())
    
    # Cek awal sebelum masuk ke library wordcloud
    if not text.strip() or len(text.split()) < 1: 
        return None
        
    try:
        wc = WordCloud(
            width=800, height=400, 
            background_color='white', 
            colormap=color,
            stopwords=custom_stopwords,
            collocations=False
        ).generate(text)
        return wc
    except ValueError:
        return None
    except Exception:
        return None

# DASHBOARD
if st.session_state.hasil_analisis is not None:
    df = st.session_state.hasil_analisis.copy()

    df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
    df['tahun'] = df['tanggal'].dt.year

    st.markdown("### Pengaturan Dashboard")
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        list_all_akun = sorted(df['sumber_akun'].unique().tolist())
        
        exclude = ["hitsunikomradio", "penerimaan_mahasiswabaru", "utv.id"]
        
        default_akun = [acc for acc in list_all_akun if acc not in exclude]
        
        sel_akun = st.multiselect(
            "Pilih Unit/Akun:", 
            options=list_all_akun, 
            default=default_akun
        )
    
    with f_col2:
        list_tahun = sorted([int(y) for y in df['tahun'].dropna().unique()])
        sel_tahun = st.multiselect("Pilih Tahun:", options=list_tahun, default=list_tahun)

    # Terapkan Filter
    df_filtered = df[df['sumber_akun'].isin(sel_akun)]
    if sel_tahun:
        df_filtered = df_filtered[df_filtered['tahun'].isin(sel_tahun)]

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "Reputasi Sentimen Per Unit", 
        "Word Cloud Bahasa", 
        "Kesimpulan & Rekomendasi"
    ])

    # TAB 1: Analisis Perbandingan
    with tab1:
        st.subheader("Analisis Perbandingan Reputasi Antar Unit")
        if not df_filtered.empty:
            df_rank = df_filtered.groupby('sumber_akun').size().reset_index(name='total_vol')
            akun_sorted = df_rank.sort_values('total_vol', ascending=False)['sumber_akun'].tolist()

            # Gunakan sistem kolom (misal: 2 kolom agar grafik tetap terbaca jelas)
            cols = st.columns(2) 

            for i, akun in enumerate(akun_sorted):
                # Masukkan ke kolom kiri atau kanan secara bergantian
                with cols[i % 2]:
                    with st.container(border=True): # Memberi bingkai agar antar unit terpisah rapi
                        df_akun = df_filtered[df_filtered['sumber_akun'] == akun]
                        
                        pos = (df_akun['sentimen'] == 'positive').sum()
                        neg = (df_akun['sentimen'] == 'negative').sum()
                        nss_val = (pos - neg) / len(df_akun) if len(df_akun) > 0 else 0
                        
                        color = "#2ecc71" if nss_val > 0 else "#e74c3c"
                        
                        # Header yang lebih ramping
                        st.markdown(f"**{akun.upper()}**")
                        st.markdown(f"<h3 style='color:{color}; margin-top:-15px;'>NSS: {nss_val:.2f}</h3>", unsafe_allow_html=True)
                        
                        df_bar = df_akun.groupby(['sentimen']).size().reset_index(name='jumlah')
                        fig_bar = px.bar(
                            df_bar, x='sentimen', y='jumlah', color='sentimen',
                            text='jumlah', height=250, # Batasi tinggi grafik agar ringkas
                            color_discrete_map={'positive':'#2ecc71', 'negative':'#e74c3c', 'neutral':'#95a5a6'}
                        )
                        fig_bar.update_traces(textposition='outside')
                        # Sembunyikan legenda untuk menghemat ruang jika sudah jelas dari sumbu X
                        fig_bar.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
                        
                        st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{akun}")

    # TAB 2
    with tab2:
        st.subheader("Eksplorasi Topik Dominan Per Unit")
        if not df_filtered.empty:
            for akun in akun_sorted:
                st.markdown(f"### Akun: {akun}")
                df_akun_wc = df_filtered[df_filtered['sumber_akun'] == akun]
                
                wc_col1, wc_col2 = st.columns(2)
                with wc_col1:
                    st.caption(f"Positif - {akun}")
                    wc_p = generate_wc(df_akun_wc[df_akun_wc['sentimen'] == 'positive']['teks_final'], 'Greens')
                    if wc_p:
                        fig, ax = plt.subplots(); ax.imshow(wc_p, interpolation='bilinear'); ax.axis('off')
                        st.pyplot(fig)
                    else: st.info("Data positif tidak mencukupi.")

                with wc_col2:
                    st.caption(f"Negatif - {akun}")
                    wc_n = generate_wc(df_akun_wc[df_akun_wc['sentimen'] == 'negative']['teks_final'], 'Reds')
                    if wc_n:
                        fig, ax = plt.subplots(); ax.imshow(wc_n, interpolation='bilinear'); ax.axis('off')
                        st.pyplot(fig)
                    else: st.info("Data negatif tidak mencukupi.")
                st.markdown("---")

    # TAB 3
    with tab3:
        st.header("Rangkuman Strategis & Wisdom")
        
        # Penjelasan NSS dengan Tombol Help (Popover)
        with st.popover("ℹ️ Apa itu NSS? (Penjelasan Metrik)"):
            st.markdown("""
            **Net Sentiment Score (NSS)** adalah metrik standar untuk mengukur keseimbangan persepsi publik.
            - **Formula:** $NSS = \\frac{\\sum Positif - \\sum Negatif}{\\sum Total}$
            - **Rentang:** -1.00 (Kritis) hingga +1.00 (Sempurna).
            - **Indikator:** Skor di atas **0.00** menunjukkan reputasi yang sehat/stabil.
            """)

        st.subheader("1. Distribusi Kesehatan Reputasi Keseluruhan")
        if not df_filtered.empty:
            df_summary = df_filtered.groupby('sumber_akun').agg(
                p=('sentimen', lambda x: (x == 'positive').sum()),
                n=('sentimen', lambda x: (x == 'negative').sum()),
                t=('sentimen', 'count')
            ).reset_index()
            df_summary['nss'] = (df_summary['p'] - df_summary['n']) / df_summary['t']
            df_summary['cat'] = df_summary['nss'].apply(lambda x: 'Reputasi Positif' if x > 0 else 'Reputasi Negatif')
            df_summary['label'] = df_summary.apply(lambda x: f"{x['sumber_akun']} (NSS: {x['nss']:.2f})", axis=1)

            fig_donut = px.sunburst(
                df_summary, path=['cat', 'label'], values='t',
                color='cat', color_discrete_map={'Reputasi Positif':'#2ecc71', 'Reputasi Negatif':'#e74c3c'},
                title="Ringkasan NSS Seluruh Unit"
            )
            fig_donut.update_traces(textinfo="label+percent entry")
            st.plotly_chart(fig_donut, use_container_width=True)

            # Metrics Best & Worst
            best_unit = df_summary.loc[df_summary['nss'].idxmax()]
            worst_unit = df_summary.loc[df_summary['nss'].idxmin()]
            
            # Global NSS
            total_f = len(df_filtered); pos_f = (df_filtered['sentimen'] == 'positive').sum(); neg_f = (df_filtered['sentimen'] == 'negative').sum()
            global_nss = (pos_f - neg_f) / total_f if total_f > 0 else 0

            c1, c2 = st.columns(2)
            c1.success(f"**Unit Terbaik:** {best_unit['sumber_akun']} ({best_unit['nss']:.2f})")
            c2.error(f"**Unit Kritis:** {worst_unit['sumber_akun']} ({worst_unit['nss']:.2f})")
            st.info(f"**Wawasan:** Secara keseluruhan, reputasi digital pada tahun {', '.join(map(str, sel_tahun))} berada dalam kondisi {'Stabil' if global_nss > 0 else 'Kritis'}.")

        st.markdown("---")
        st.subheader("2. Pola Topik & Bahasa Dominan")
        
        def get_top_keywords(data_teks, n=5):
            noise_words = [
                'dan', 'yang', 'di', 'ke', 'dari', 'itu', 'ini', 'untuk', 'dengan', 
                'ada', 'aja', 'ya', 'ga', 'gak', 'unikom', 'banget', 'bgt', 'kita', 
                'adalah', 'akan', 'bisa', 'sudah', 'kalo', 'kalau', 'paling', 'anak',
                'akang', 'teteh', 'kak', 'bang', 'admin', 'min', 'sama', 'saya',
                'tapi', 'begitu', 'begini', 'gitu', 'gini', 'seperti', 'malah',
                'buat', 'nanti', 'bakal', 'pernah', 'kok', 'nih', 'sih', 'tuh', 
                'deh', 'dong', 'lho', 'kah', 'lah', 'cuma', 'hanya', 'seru', 'wleu',
                'kali', 'bu', 'gh', 'nurullss', 'ey', 'igin'
            ]
            
            interest_topics = [
                'ukt', 'mahal', 'biaya', 'dosen', 'parkir', 'pendaftaran', 'kuliah',
                'beasiswa', 'wisuda', 'skripsi', 'fasilitas', 'gedung', 'pcr',
                'registrasi', 'akreditasi', 'prestasi', 'juara', 'lomba', 'kurang'
            ]
            
            # Gabungkan dan bersihkan teks
            words = " ".join(data_teks.dropna().astype(str)).lower().split()
            
            # Filter kata: bukan noise, panjang > 3, atau masuk dalam interest_topics
            filtered_words = [
                w for w in words 
                if (w not in noise_words and len(w) > 3) or (w in interest_topics)
            ]
            
            if not filtered_words:
                return ["Tidak ada topik dominan"]
                
            # Hitung frekuensi
            top_counts = pd.Series(filtered_words).value_counts()
            
            # Prioritaskan kata yang ada di interest_topics jika frekuensinya cukup
            return top_counts.head(n).index.tolist()

        t_pos = get_top_keywords(df_filtered[df_filtered['sentimen'] == 'positive']['teks_final'])
        t_neg = get_top_keywords(df_filtered[df_filtered['sentimen'] == 'negative']['teks_final'])
        
        st.write(f"- **Pemicu Positif:** {', '.join(t_pos) if t_pos else 'Data tidak cukup'}")
        st.write(f"- **Pemicu Negatif:** {', '.join(t_neg) if t_neg else 'Data tidak cukup'}")

        # --- 3. Rekomendasi Strategis & Implementasi SOP (Jawaban Tujuan 1.3) ---
        st.markdown("---")
        st.subheader("Rekomendasi Strategis (SOP)")
        
        # Logika dinamis untuk menentukan pesan berdasarkan data
        top_pos = t_pos[0] if t_pos else "prestasi/fasilitas"
        top_neg = t_neg[0] if t_neg else "layanan/biaya"

        st.markdown(f"""
        Berdasarkan **Analisis Data**, berikut adalah langkah strategis yang direkomendasikan untuk Divisi Digital Marketing:

        **A. Manajemen Reputasi Unit:**
        1. **Benchmarking:** Menetapkan unit `{best_unit['sumber_akun']}` sebagai standar kualitas karena berhasil menjaga skor NSS di angka `{best_unit['nss']:.2f}`. Gaya bahasa dan interaksi pada unit ini harus dipelajari oleh unit lain.
        2. **Intervensi Prioritas:** Melakukan pendampingan khusus pada unit `{worst_unit['sumber_akun']}`. Tim pusat perlu melakukan audit terhadap gaya admin dalam pembuatan konten agar lebih cocok dengan audiens.

        **B. Strategi Konten & Mitigasi:**
        1. **Amplifikasi Sentimen Positif:** Meningkatkan feedback positif dengan merancang konten yang berfokus pada pola bahasa terkait **"{top_pos}"**. Topik ini terbukti menjadi pemicu kepuasan utama audiens.
        2. **Mitigasi Krisis & Respon:** Menyusun draf FAQ atau narasi klarifikasi resmi terkait topik **"{top_neg}"**. Hal ini diperlukan untuk mengurangi eskalasi komentar negatif yang dominan muncul pada platform media sosial UNIKOM terkait isu tersebut.
        """)

    # Download Data Hasil Analisis
    st.markdown("---")
    @st.cache_data
    def convert_df(df): return df.to_csv(index=False).encode('utf-8')
    csv_data = convert_df(df_filtered)
    st.download_button("💾 Unduh Data Hasil Analisis (.csv)", csv_data, "unikom_sentiment_filtered.csv", "text/csv")