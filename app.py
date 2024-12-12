import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Judul
st.title("Dasbor Integrasi Pasca-Merger Pelindo")
st.write("Aplikasi ini menyediakan wawasan dan alat untuk proses integrasi pasca-merger Pelindo.")

# Upload file oleh pengguna
uploaded_file = st.file_uploader("Unggah file Excel atau CSV Anda", type=["xlsx", "csv"])

if uploaded_file:
    # Memuat data dari file yang diunggah
    if uploaded_file.name.endswith("xlsx"):
        excel_file = pd.ExcelFile(uploaded_file)
        sheet = st.sidebar.selectbox("Pilih Sheet", excel_file.sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet)
    elif uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
        sheet = "CSV File"

    # Tampilkan data
    st.subheader(f"Data dari {sheet}")
    st.dataframe(df)

    # Analisis khusus berdasarkan pilihan sheet
    if sheet == "Daftar IS & AK":
        st.subheader("Analisis Inisiatif Strategis dan Aksi Kunci")
        st.write("Menganalisis hubungan dan prioritas antara IS dan AK.")
        if "Priority" in df.columns:
            st.write("Prioritas Utama:")
            st.write(df[df["Priority"] == "High"])

    elif sheet == "Evaluasi Target":
        st.subheader("Evaluasi Target")
        st.write("Ikhtisar pencapaian target dan kesenjangan.")
        if "Achievement" in df.columns and "Target" in df.columns:
            df['Gap'] = df['Target'] - df['Achievement']
            st.write("Kesenjangan Kinerja:")
            st.dataframe(df[['Target', 'Achievement', 'Gap']])
            # Visualisasi lanjutan
            st.write("Distribusi Kesenjangan:")
            fig, ax = plt.subplots()
            sns.histplot(df['Gap'], kde=True, ax=ax)
            ax.set_title("Distribusi Kesenjangan Kinerja")
            st.pyplot(fig)

    elif "Dashboard" in sheet:
        st.subheader(f"Analisis {sheet}")
        st.write("Visualisasi metrik utama dan tren.")
        numeric_data = df.select_dtypes(include=['float', 'int'])
        if not numeric_data.empty:
            st.line_chart(numeric_data)
            st.write("Matriks Korelasi:")
            corr = numeric_data.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Matriks Korelasi Data Numerik")
            st.pyplot(fig)
        else:
            st.write("Tidak ada data numerik yang tersedia untuk visualisasi.")

    elif sheet == "Tracking VC":
        st.subheader("Pelacakan Penciptaan Nilai")
        st.write("Memantau kemajuan dan mengidentifikasi hambatan.")
        if "Progress" in df.columns:
            avg_progress = df["Progress"].mean()
            st.metric("Rata-rata Kemajuan", f"{avg_progress:.2f}%")
            st.write("Distribusi Kemajuan:")
            fig, ax = plt.subplots()
            sns.boxplot(x=df['Progress'], ax=ax)
            ax.set_title("Distribusi Kemajuan")
            st.pyplot(fig)

    elif sheet in ["RKM 2", "HO", "SPTP", "SPMT", "SPSL", "SPJM"]:
        st.subheader(f"Wawasan Operasional untuk {sheet}")
        st.write("Menganalisis data operasional dan mengidentifikasi area untuk perbaikan.")
        st.write("Statistik Ringkasan:")
        st.write(df.describe())
        # Visualisasi lanjutan
        numeric_data = df.select_dtypes(include=['float', 'int'])
        if not numeric_data.empty:
            st.write("Distribusi Data:")
            fig, ax = plt.subplots(figsize=(10, 6))
            numeric_data.plot(kind='box', ax=ax)
            ax.set_title(f"Distribusi Data Numerik untuk {sheet}")
            st.pyplot(fig)

    elif sheet == "99 Ref":
        st.subheader("Analisis Data Referensi")
        st.write("Detail dan wawasan dari data referensi.")
        st.write(df)

    # Integrasi GPT-4o
    st.subheader("Analisis AI (GPT-4o)")
    if os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        prompt = st.text_area("Ajukan pertanyaan kepada GPT-4o tentang data ini", "Apa wawasan yang bisa Anda berikan tentang dataset ini?")
        if st.button("Analisis"):
            with st.spinner("GPT-4o sedang menganalisis data..."):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "Anda adalah asisten yang menganalisis data."},
                            {"role": "user", "content": f"Analisis dataset berikut:\n{df.head(10).to_string()}\n\n{prompt}"}
                        ],
                        max_tokens=500
                    )
                    st.write(response['choices'][0]['message']['content'])
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
    else:
        st.info("Masukkan API Key OpenAI Anda di file .env untuk mengaktifkan analisis GPT-4o.")
else:
    st.warning("Harap unggah file untuk memulai.")
