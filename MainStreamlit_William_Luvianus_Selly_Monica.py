import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import os

# ── Cari model di beberapa lokasi ──────────────────────────────────────────────
def _find_model_file(filename):
    """Coba beberapa lokasi umum untuk menemukan file pkl."""
    candidates = [
        filename,
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.getcwd(), filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return filename  # kembalikan nama asli agar pesan error lebih jelas

CLF_MODEL_PATH = _find_model_file("BestModel_Klasifikasi_Alg_William_Luvianus_Selly_Monica.pkl")
REG_MODEL_PATH = _find_model_file("BestModel_Regresi_Alg_William_Luvianus_Selly_Monica.pkl")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Gaya Belajar & IPK",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Font & background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #1a2e4a 0%, #0f1f33 100%);
    }
    section[data-testid="stSidebar"] * { color: #e8edf2 !important; }
    section[data-testid="stSidebar"] .stRadio label { color: #c9d6e3 !important; }

    /* Cards */
    .result-card {
        background: linear-gradient(135deg, #1a2e4a 0%, #163348 100%);
        border-left: 5px solid #4fc3f7;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 16px 0;
        color: #e8edf2;
    }
    .result-card h2 { color: #4fc3f7; margin: 0 0 4px 0; font-size: 2rem; }
    .result-card p  { color: #a8bcc9; margin: 0; font-size: 0.9rem; }

    .info-card {
        background: #f0f7ff;
        border-left: 4px solid #1976d2;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.9rem;
        color: #1a3a5c;
    }

    .vark-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 6px;
    }
    .badge-visual      { background: #e3f2fd; color: #1565c0; }
    .badge-auditory    { background: #f3e5f5; color: #6a1b9a; }
    .badge-readwrite   { background: #e8f5e9; color: #2e7d32; }
    .badge-kinesthetic { background: #fff3e0; color: #e65100; }

    /* Metric chips */
    .chip-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; }
    .chip {
        background: #e8edf2;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.82rem;
        color: #1a2e4a;
        font-weight: 500;
    }

    /* Section header */
    .section-header {
        border-bottom: 2px solid #e0e8f0;
        padding-bottom: 6px;
        margin-bottom: 16px;
        color: #1a2e4a;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #8899aa;
        font-size: 0.78rem;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #e0e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    errors = {}

    def load_model(path, label):
        if not os.path.exists(path):
            errors[label] = (
                f"File model {label} tidak ditemukan: {path}\n"
                f"   Jalankan notebook terkait sampai cell export model, atau pastikan file .pkl ada di folder aplikasi."
            )
            return None
        try:
            return joblib.load(path)
        except Exception as exc:
            errors[label] = (
                f"Gagal load model {label} ({path}): {type(exc).__name__}: {exc}\n"
                f"   scikit-learn yang berjalan saat ini: {sklearn.__version__}\n"
                f"   Kemungkinan besar versi scikit-learn saat training (notebook) "
                f"berbeda dengan versi di requirements.txt Streamlit Cloud.\n"
                f"   Solusi: cocokkan scikit-learn=={sklearn.__version__} di requirements.txt, "
                f"atau retrain model dengan versi yang sama dengan requirements.txt."
            )
            return None

    clf = load_model(CLF_MODEL_PATH, "klasifikasi")
    reg = load_model(REG_MODEL_PATH, "regresi")
    return clf, reg, errors

clf_model, reg_model, model_load_errors = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Menu Utama")
    menu = st.radio(
        "Pilih fitur:",
        ["🔍 Klasifikasi Gaya Belajar", "📈 Prediksi IPK"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Tentang Aplikasi**")
    st.markdown(
        "Aplikasi ini menggunakan model Machine Learning "
        "untuk memprediksi gaya belajar dan IPK mahasiswa "
        "berdasarkan survei kebiasaan belajar."
    )
    st.markdown("---")
    st.markdown("**👥 Kelompok**")
    st.markdown("William Luvianus · 230712339")
    st.markdown("Selly Monica · 230712375")
    st.markdown("---")
    st.markdown("**Mata Kuliah**")
    st.markdown("Pembelajaran Mesin · UAS 2025/2026")
    st.markdown("Informatika — UAJY")
    st.markdown("---")
    with st.expander("⚙️ Info Versi (debug)"):
        st.caption(f"scikit-learn: {sklearn.__version__}")
        st.caption(f"pandas: {pd.__version__}")
        st.caption(f"numpy: {np.__version__}")
        st.caption(f"joblib: {joblib.__version__}")

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: shared demographic inputs
# ══════════════════════════════════════════════════════════════════════════════
def input_demografi(prefix=""):
    col1, col2 = st.columns(2)
    with col1:
        usia = st.number_input("Usia (tahun)", min_value=17, max_value=35,
                               value=21, key=f"{prefix}_usia")
        semester = st.selectbox("Semester saat ini",
                                [1, 2, 3, 4, 5, 6, 7, "8 atau lebih"],
                                index=4, key=f"{prefix}_semester")
        jenis_kelamin = st.selectbox("Jenis Kelamin",
                                     ["Perempuan", "Laki-laki"],
                                     key=f"{prefix}_gender")
        status_tinggal = st.selectbox(
            "Status Tempat Tinggal",
            ["Kost/asrama", "Bersama orang tua",
             "Kontrak/tinggal mandiri", "Wali"],
            key=f"{prefix}_tinggal",
        )
    with col2:
        kerja = st.selectbox("Apakah bekerja sambil kuliah?",
                             ["Tidak", "Ya, paruh waktu", "Ya, penuh waktu"],
                             key=f"{prefix}_kerja")
        waktu_belajar = st.selectbox(
            "Waktu belajar mandiri/hari",
            ["Kurang dari 1 jam", "1–2 jam", "2–3 jam",
             "3–4 jam", "Lebih dari 4 jam"],
            index=1, key=f"{prefix}_waktu",
        )
        internet = st.slider("Kualitas akses internet (1–5)",
                             1, 5, 3, key=f"{prefix}_internet")
        kehadiran = st.slider("Tingkat kehadiran kuliah (1–5)",
                              1, 5, 4, key=f"{prefix}_hadir")
        perangkat = st.selectbox(
            "Perangkat utama belajar",
            ["Laptop", "Smartphone", "Laptop, Smartphone",
             "Laptop, Tablet", "Laptop, Smartphone, Tablet",
             "Komputer desktop"],
            index=2, key=f"{prefix}_perangkat",
        )
    return dict(
        usia=usia, semester=semester,
        jenis_kelamin=jenis_kelamin, status_tinggal=status_tinggal,
        kerja=kerja, waktu_belajar=waktu_belajar,
        internet=internet, kehadiran=kehadiran, perangkat=perangkat,
    )


def input_likert_block(label, keys, prefix, n=5):
    """Render n Likert sliders and return list of values."""
    vals = []
    for i, (lbl, key) in enumerate(zip(label, keys)):
        v = st.slider(f"{i+1}. {lbl}", 1, 5, 3, key=f"{prefix}_{key}")
        vals.append(v)
    return vals


def build_feature_row(demo, d_vals, e_vals, f_vals, learning_style=None,
                      ips=None, mengulang=None, nilai_de=None):
    """Assemble a 1-row DataFrame matching the pipeline's expected columns."""

    # ── Kolom nama persis seperti di dataset ──────────────────────────────────
    row = {
        ' Usia  (tahun)'                                                        : demo['usia'],
        ' Kualitas akses internet untuk belajar  '                              : demo['internet'],
        ' Kehadiran perkuliahan saya selama satu semester terakhir secara umum  ': demo['kehadiran'],
        ' Semester saat ini  '                                                  : 8 if demo['semester'] == "8 atau lebih" else int(demo['semester']),
        ' Jenis kelamin  '                                                      : demo['jenis_kelamin'],
        ' Status tempat tinggal selama kuliah  '                                : demo['status_tinggal'],
        ' Apakah Anda bekerja sambil kuliah?  '                                 : demo['kerja'],
        ' Rata-rata waktu belajar mandiri per hari di luar jam kuliah  '        : demo['waktu_belajar'],
        ' Perangkat utama yang digunakan untuk belajar  '                       : demo['perangkat'],
    }

    # ── Kolom D (Kebiasaan Belajar) 20 item ──────────────────────────────────
    d_col_names = [
        ' Saya memiliki jadwal belajar yang cukup teratur setiap minggu.  ',
        'Saya membagi waktu belajar jauh sebelum ujian atau tenggat tugas. ',
        'Saya mampu menentukan prioritas antara kuliah, tugas, dan kegiatan lain. ',
        'Saya jarang menunda mengerjakan tugas sampai mendekati tenggat waktu. ',
        'Saya konsisten mengalokasikan waktu khusus untuk belajar mandiri. ',
        'Saya meninjau kembali materi kuliah setelah pertemuan selesai. ',
        'Saya mencari sumber tambahan ketika ada materi yang belum saya pahami. ',
        'Saya membandingkan berbagai sumber untuk memahami suatu topik. ',
        'Saya membuat ringkasan, poin penting, atau catatan pribadi dari materi kuliah. ',
        'Saya mengecek sendiri apakah saya benar-benar sudah memahami materi. ',
        'Saya memiliki target akademik yang jelas setiap semester. ',
        'Saya tetap berusaha memahami materi walaupun terasa sulit. ',
        'Saya terdorong untuk belajar bukan hanya demi nilai, tetapi juga demi pemahaman. ',
        'Saya merasa bertanggung jawab terhadap hasil belajar saya sendiri. ',
        'Saya berusaha memperbaiki strategi belajar ketika hasil saya kurang baik. ',
        'Saya aktif bertanya atau berdiskusi ketika ada materi yang belum jelas. ',
        'Saya mengerjakan tugas kuliah dengan sungguh-sungguh. ',
        'Saya mengikuti perkuliahan dengan fokus. ',
        'Saya berusaha hadir tepat waktu dalam perkuliahan. ',
        'Saya memanfaatkan umpan balik dari dosen untuk memperbaiki hasil belajar saya. ',
    ]
    for col, val in zip(d_col_names, d_vals):
        row[col] = val

    # ── Kolom E (Lingkungan) 5 item ───────────────────────────────────────────
    e_col_names = [
        'Saya memiliki tempat belajar yang cukup nyaman. ',
        'Kondisi lingkungan tempat saya belajar mendukung konsentrasi saya. ',
        'Saya memiliki akses perangkat yang memadai untuk mengerjakan tugas kuliah. ',
        'Saya memiliki dukungan sosial yang cukup dari keluarga atau teman untuk belajar. ',
        'Hambatan teknis seperti internet atau perangkat sering mengganggu proses belajar saya. ',
    ]
    for col, val in zip(e_col_names, e_vals):
        row[col] = val

    # ── Kolom F (Teknologi) 5 item ────────────────────────────────────────────
    f_col_names = [
        'Saya memanfaatkan platform digital untuk memahami materi kuliah. ',
        'Video pembelajaran atau tutorial online membantu saya memahami materi. ',
        'Saya menggunakan aplikasi atau tools digital untuk mencatat, merangkum, atau mengatur belajar. ',
        'Saya dapat belajar secara efektif melalui media pembelajaran online. ',
        'Saya mudah terdistraksi oleh penggunaan perangkat digital saat sedang belajar. ',
    ]
    for col, val in zip(f_col_names, f_vals):
        row[col] = val

    # ── Kolom regresi-only ────────────────────────────────────────────────────
    if learning_style is not None:
        row['LearningStyle'] = learning_style
    if ips is not None:
        row[' IPS semester terakhir  '] = ips
    if mengulang is not None:
        row['Dalam satu semester terakhir, apakah Anda pernah mengulang mata kuliah?  '] = mengulang
    if nilai_de is not None:
        row['Dalam satu semester terakhir, apakah Anda pernah mendapat nilai D/E? '] = nilai_de

    return pd.DataFrame([row])


def align_columns_to_model(df_input, model):
    """
    Coba sesuaikan nama kolom df_input dengan yang diharapkan pipeline model.
    Jika pipeline punya ColumnTransformer, ambil daftar kolom dari sana.
    Fallback: normalisasi whitespace dengan strip().
    """
    try:
        # Ambil expected columns dari ColumnTransformer di dalam pipeline
        ct = model.named_steps.get('prep')
        if ct is None:
            return df_input
        expected_cols = []
        for _, _, cols in ct.transformers:
            if isinstance(cols, (list, tuple, pd.Index, np.ndarray)):
                expected_cols.extend(cols)
        if not expected_cols:
            return df_input

        # Buat mapping: stripped_name -> original_expected_name
        strip_map = {c.strip(): c for c in expected_cols}

        rename_map = {}
        for col in df_input.columns:
            stripped = col.strip()
            if col not in expected_cols and stripped in strip_map:
                rename_map[col] = strip_map[stripped]

        if rename_map:
            df_input = df_input.rename(columns=rename_map)

        # Tambah kolom yang hilang dengan NaN
        for col in expected_cols:
            if col not in df_input.columns:
                df_input[col] = np.nan
    except Exception:
        pass  # Jika gagal, biarkan model handle sendiri
    return df_input



# TAB 1 — KLASIFIKASI GAYA BELAJAR
# ══════════════════════════════════════════════════════════════════════════════
if menu == "🔍 Klasifikasi Gaya Belajar":
    st.title("🔍 Klasifikasi Gaya Belajar Mahasiswa")
    st.markdown(
        "Isi survei di bawah ini untuk mengetahui gaya belajar dominan kamu "
        "berdasarkan model Machine Learning (VARK Framework)."
    )

    if clf_model is None:
        st.error(f"⚠️ Model klasifikasi belum bisa digunakan. "
                 f"Pastikan `{CLF_MODEL_PATH}` ada dan cocok dengan versi dependency.")
        if "klasifikasi" in model_load_errors:
            st.code(model_load_errors["klasifikasi"], language="text")
        st.stop()

    with st.form("form_klasifikasi"):
        # ── Demografi ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-header">📋 Data Diri</p>',
                    unsafe_allow_html=True)
        demo = input_demografi("clf")

        st.markdown('<p class="section-header">🏆 Informasi Akademik</p>',
                    unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            mengulang_clf = st.selectbox("Pernah mengulang mata kuliah?",
                                         ["Tidak", "Ya"], key="clf_mengulang")
        with col_b:
            nilai_de_clf = st.selectbox("Pernah mendapat nilai D/E?",
                                        ["Tidak", "Ya"], key="clf_nilaide")

        # ── Kebiasaan Belajar D ───────────────────────────────────────────────
        st.markdown('<p class="section-header">📚 Kebiasaan Belajar (skala 1–5: 1=Tidak pernah, 5=Selalu)</p>',
                    unsafe_allow_html=True)

        d_labels = [
            "Saya memiliki jadwal belajar yang teratur setiap minggu",
            "Saya membagi waktu belajar jauh sebelum ujian",
            "Saya mampu menentukan prioritas kuliah, tugas, dan kegiatan lain",
            "Saya jarang menunda tugas sampai mendekati tenggat",
            "Saya konsisten mengalokasikan waktu untuk belajar mandiri",
            "Saya meninjau kembali materi setelah kuliah selesai",
            "Saya mencari sumber tambahan saat materi belum dipahami",
            "Saya membandingkan berbagai sumber untuk memahami topik",
            "Saya membuat ringkasan atau catatan pribadi dari materi",
            "Saya mengecek sendiri apakah sudah benar-benar paham",
            "Saya memiliki target akademik yang jelas setiap semester",
            "Saya tetap berusaha memahami materi walaupun sulit",
            "Saya belajar demi pemahaman, bukan hanya nilai",
            "Saya merasa bertanggung jawab terhadap hasil belajar sendiri",
            "Saya memperbaiki strategi belajar saat hasil kurang baik",
            "Saya aktif bertanya saat ada materi yang belum jelas",
            "Saya mengerjakan tugas kuliah dengan sungguh-sungguh",
            "Saya mengikuti perkuliahan dengan fokus",
            "Saya berusaha hadir tepat waktu",
            "Saya memanfaatkan umpan balik dosen untuk perbaikan",
        ]
        d_keys = [f"d{i}" for i in range(20)]
        cols_d = st.columns(2)
        d_vals = []
        for i, (lbl, key) in enumerate(zip(d_labels, d_keys)):
            with cols_d[i % 2]:
                d_vals.append(st.slider(f"{i+1}. {lbl}", 1, 5, 3,
                                        key=f"clf_{key}"))

        # ── Lingkungan E ──────────────────────────────────────────────────────
        st.markdown('<p class="section-header">🏠 Lingkungan Belajar</p>',
                    unsafe_allow_html=True)
        e_labels = [
            "Saya memiliki tempat belajar yang nyaman",
            "Lingkungan belajar saya mendukung konsentrasi",
            "Saya memiliki akses perangkat yang memadai",
            "Saya memiliki dukungan sosial yang cukup dari keluarga/teman",
            "Hambatan teknis (internet/perangkat) sering mengganggu belajar saya ⟵ (item balik)",
        ]
        e_keys = [f"e{i}" for i in range(5)]
        cols_e = st.columns(2)
        e_vals = []
        for i, (lbl, key) in enumerate(zip(e_labels, e_keys)):
            with cols_e[i % 2]:
                e_vals.append(st.slider(f"{i+1}. {lbl}", 1, 5, 3,
                                        key=f"clf_{key}"))

        # ── Teknologi F ───────────────────────────────────────────────────────
        st.markdown('<p class="section-header">💻 Penggunaan Teknologi Belajar</p>',
                    unsafe_allow_html=True)
        f_labels = [
            "Saya memanfaatkan platform digital untuk memahami materi",
            "Video/tutorial online membantu saya memahami materi",
            "Saya pakai aplikasi digital untuk mencatat/merangkum",
            "Saya bisa belajar efektif melalui media online",
            "Saya mudah terdistraksi perangkat digital saat belajar ⟵ (item balik)",
        ]
        f_keys = [f"f{i}" for i in range(5)]
        cols_f = st.columns(2)
        f_vals = []
        for i, (lbl, key) in enumerate(zip(f_labels, f_keys)):
            with cols_f[i % 2]:
                f_vals.append(st.slider(f"{i+1}. {lbl}", 1, 5, 3,
                                        key=f"clf_{key}"))

        submitted = st.form_submit_button("🔍 Prediksi Gaya Belajar",
                                          use_container_width=True,
                                          type="primary")

    # ── Prediksi ──────────────────────────────────────────────────────────────
    if submitted:
        # Reverse coding E5, F5 sebelum masuk model
        e_vals_rc = e_vals.copy()
        e_vals_rc[4] = 6 - e_vals_rc[4]
        f_vals_rc = f_vals.copy()
        f_vals_rc[4] = 6 - f_vals_rc[4]

        X_input = build_feature_row(demo, d_vals, e_vals_rc, f_vals_rc)

        # Fitur ini dipakai di notebook klasifikasi, jadi nilainya diambil dari form.
        X_input['Dalam satu semester terakhir, apakah Anda pernah mengulang mata kuliah?  '] = mengulang_clf
        X_input['Dalam satu semester terakhir, apakah Anda pernah mendapat nilai D/E? '] = nilai_de_clf

        try:
            X_input = align_columns_to_model(X_input, clf_model)
            pred = clf_model.predict(X_input)[0]
            proba = clf_model.predict_proba(X_input)[0]
            classes = clf_model.classes_

            # ── Badge warna per gaya belajar ──────────────────────────────────
            badge_class = {
                "Visual"     : "badge-visual",
                "Auditory"   : "badge-auditory",
                "ReadWrite"  : "badge-readwrite",
                "Kinesthetic": "badge-kinesthetic",
            }
            icon_map = {
                "Visual": "👁️", "Auditory": "👂",
                "ReadWrite": "📖", "Kinesthetic": "🤸",
            }
            desc_map = {
                "Visual"     : "Kamu belajar paling efektif melalui diagram, bagan, warna, dan tampilan visual.",
                "Auditory"   : "Kamu belajar paling efektif melalui diskusi, penjelasan lisan, dan rekaman audio.",
                "ReadWrite"  : "Kamu belajar paling efektif melalui membaca, menulis catatan, dan merangkum.",
                "Kinesthetic": "Kamu belajar paling efektif melalui praktik langsung, simulasi, dan contoh nyata.",
            }

            st.markdown("---")
            st.markdown("### ✅ Hasil Prediksi")

            st.markdown(f"""
            <div class="result-card">
                <p>Gaya Belajar Dominan</p>
                <h2>{icon_map[pred]} {pred}</h2>
                <span class="vark-badge {badge_class[pred]}">{pred} Learner</span>
                <p style="margin-top:10px;">{desc_map[pred]}</p>
            </div>
            """, unsafe_allow_html=True)

            # Probabilitas semua kelas
            st.markdown("**Probabilitas per Gaya Belajar:**")
            prob_df = pd.DataFrame({
                "Gaya Belajar": classes,
                "Probabilitas": proba,
            }).sort_values("Probabilitas", ascending=False)

            for _, row in prob_df.iterrows():
                st.progress(float(row["Probabilitas"]),
                            text=f"{row['Gaya Belajar']}: {row['Probabilitas']:.1%}")

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
            st.info("Pastikan kolom input sesuai dengan yang digunakan saat training model.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDIKSI IPK
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("📈 Prediksi IPK Mahasiswa")
    st.markdown(
        "Isi survei di bawah ini untuk memprediksi IPK kumulatif kamu "
        "berdasarkan gaya belajar dan kebiasaan akademik."
    )

    if reg_model is None:
        st.error(f"⚠️ Model regresi belum bisa digunakan. "
                 f"Pastikan `{REG_MODEL_PATH}` ada dan cocok dengan versi dependency.")
        if "regresi" in model_load_errors:
            st.code(model_load_errors["regresi"], language="text")
        st.stop()

    with st.form("form_regresi"):
        # ── Demografi ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-header">📋 Data Diri</p>',
                    unsafe_allow_html=True)
        demo = input_demografi("reg")

        # ── Akademik tambahan ─────────────────────────────────────────────────
        st.markdown('<p class="section-header">🏆 Informasi Akademik</p>',
                    unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            ips = st.number_input("IPS Semester Terakhir", min_value=0.0,
                                  max_value=4.0, value=3.5, step=0.01,
                                  format="%.2f", key="reg_ips")
        with col_b:
            mengulang = st.selectbox("Pernah mengulang mata kuliah?",
                                     ["Tidak", "Ya"], key="reg_mengulang")
        with col_c:
            nilai_de = st.selectbox("Pernah mendapat nilai D/E?",
                                    ["Tidak", "Ya"], key="reg_nilaide")

        # ── Gaya Belajar ──────────────────────────────────────────────────────
        st.markdown('<p class="section-header">🎨 Gaya Belajar</p>',
                    unsafe_allow_html=True)
        learning_style = st.selectbox(
            "Gaya belajar dominan kamu (dapat diisi dari hasil Tab 1)",
            ["Visual", "Auditory", "ReadWrite", "Kinesthetic"],
            key="reg_ls",
        )
        st.markdown(
            '<div class="info-card">💡 Tidak tahu gaya belajarmu? '
            'Gunakan menu <b>Klasifikasi Gaya Belajar</b> terlebih dahulu.</div>',
            unsafe_allow_html=True,
        )

        # ── Kebiasaan Belajar D ───────────────────────────────────────────────
        st.markdown('<p class="section-header">📚 Kebiasaan Belajar (skala 1–5)</p>',
                    unsafe_allow_html=True)
        d_labels_reg = [
            "Memiliki jadwal belajar yang teratur",
            "Membagi waktu belajar jauh sebelum ujian",
            "Mampu menentukan prioritas kuliah dan kegiatan",
            "Jarang menunda tugas sampai mendekati tenggat",
            "Konsisten mengalokasikan waktu belajar mandiri",
            "Meninjau kembali materi setelah kuliah",
            "Mencari sumber tambahan saat materi belum dipahami",
            "Membandingkan berbagai sumber belajar",
            "Membuat ringkasan/catatan dari materi",
            "Mengecek pemahaman diri sendiri secara mandiri",
            "Memiliki target akademik yang jelas",
            "Tetap berusaha meski materi terasa sulit",
            "Belajar demi pemahaman, bukan hanya nilai",
            "Merasa bertanggung jawab atas hasil belajar",
            "Memperbaiki strategi belajar saat hasil kurang baik",
            "Aktif bertanya saat materi belum jelas",
            "Mengerjakan tugas dengan sungguh-sungguh",
            "Mengikuti perkuliahan dengan fokus",
            "Berusaha hadir tepat waktu",
            "Memanfaatkan umpan balik dosen",
        ]
        cols_d_r = st.columns(2)
        d_vals_r = []
        for i, lbl in enumerate(d_labels_reg):
            with cols_d_r[i % 2]:
                d_vals_r.append(st.slider(f"{i+1}. {lbl}", 1, 5, 3,
                                          key=f"reg_d{i}"))

        # ── Lingkungan E ──────────────────────────────────────────────────────
        st.markdown('<p class="section-header">🏠 Lingkungan Belajar</p>',
                    unsafe_allow_html=True)
        e_labels_reg = [
            "Memiliki tempat belajar yang nyaman",
            "Lingkungan mendukung konsentrasi",
            "Memiliki akses perangkat yang memadai",
            "Memiliki dukungan sosial yang cukup",
            "Hambatan teknis sering mengganggu belajar ⟵ (item balik)",
        ]
        cols_e_r = st.columns(2)
        e_vals_r = []
        for i, lbl in enumerate(e_labels_reg):
            with cols_e_r[i % 2]:
                e_vals_r.append(st.slider(f"{i+1}. {lbl}", 1, 5, 3,
                                          key=f"reg_e{i}"))

        # ── Teknologi F ───────────────────────────────────────────────────────
        st.markdown('<p class="section-header">💻 Penggunaan Teknologi Belajar</p>',
                    unsafe_allow_html=True)
        f_labels_reg = [
            "Memanfaatkan platform digital untuk belajar",
            "Video/tutorial online membantu pemahaman",
            "Memakai aplikasi digital untuk mencatat/merangkum",
            "Bisa belajar efektif via media online",
            "Mudah terdistraksi perangkat digital ⟵ (item balik)",
        ]
        cols_f_r = st.columns(2)
        f_vals_r = []
        for i, lbl in enumerate(f_labels_reg):
            with cols_f_r[i % 2]:
                f_vals_r.append(st.slider(f"{i+1}. {lbl}", 1, 5, 3,
                                          key=f"reg_f{i}"))

        submitted_reg = st.form_submit_button("📈 Prediksi IPK",
                                              use_container_width=True,
                                              type="primary")

    # ── Prediksi ──────────────────────────────────────────────────────────────
    if submitted_reg:
        # Reverse coding
        e_vals_rc_r = e_vals_r.copy(); e_vals_rc_r[4] = 6 - e_vals_rc_r[4]
        f_vals_rc_r = f_vals_r.copy(); f_vals_rc_r[4] = 6 - f_vals_rc_r[4]

        X_input_r = build_feature_row(
            demo, d_vals_r, e_vals_rc_r, f_vals_rc_r,
            learning_style=learning_style,
            ips=ips, mengulang=mengulang, nilai_de=nilai_de,
        )

        try:
            X_input_r = align_columns_to_model(X_input_r, reg_model)
            pred_ipk = reg_model.predict(X_input_r)[0]
            pred_ipk = float(np.clip(pred_ipk, 0.0, 4.0))

            # ── Warna berdasarkan prediksi ────────────────────────────────────
            if pred_ipk >= 3.5:
                grade, color = "Cum Laude / Sangat Memuaskan", "#4fc3f7"
            elif pred_ipk >= 3.0:
                grade, color = "Memuaskan", "#81c784"
            elif pred_ipk >= 2.5:
                grade, color = "Cukup", "#ffb74d"
            else:
                grade, color = "Perlu Peningkatan", "#e57373"

            st.markdown("---")
            st.markdown("### ✅ Hasil Prediksi")

            st.markdown(f"""
            <div class="result-card" style="border-left-color:{color}">
                <p>Prediksi IPK Kumulatif</p>
                <h2 style="color:{color}">{pred_ipk:.2f}</h2>
                <span class="vark-badge" style="background:#1e3a5f;color:{color};">
                    {grade}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # ── Progress bar ──────────────────────────────────────────────────
            st.markdown("**Posisi IPK dalam skala 0.0 – 4.0:**")
            st.progress(pred_ipk / 4.0,
                        text=f"IPK {pred_ipk:.2f} / 4.00")

            # ── Interpretasi ──────────────────────────────────────────────────
            st.markdown(f"""
            <div class="info-card">
                <b>📌 Interpretasi:</b> Berdasarkan gaya belajar <b>{learning_style}</b>,
                IPS terakhir <b>{ips:.2f}</b>, dan kebiasaan belajar yang diisi,
                model memprediksi IPK kumulatif kamu sekitar <b>{pred_ipk:.2f}</b>
                ({grade}).
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
            st.info("Pastikan kolom input sesuai dengan yang digunakan saat training model.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Projek UAS Pembelajaran Mesin · Genap T.A. 2025/2026 ·
    William Luvianus (230712339) & Selly Monica (230712375) ·
    Informatika — Universitas Atma Jaya Yogyakarta
</div>
""", unsafe_allow_html=True)
