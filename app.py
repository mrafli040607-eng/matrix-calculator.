from pathlib import Path

app_code = r'''import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import time

# =========================================================
# KONFIGURASI DAN TEMA
# =========================================================
st.set_page_config(
    page_title="MatrixLab | Kalkulator Matriks",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #07111f;
        --panel: #0d1b2d;
        --panel2: #10243a;
        --cyan: #42d9ff;
        --blue: #4388ff;
        --text: #e8f2ff;
        --muted: #9eb4cc;
    }
    .stApp {
        background:
          radial-gradient(circle at 12% 0%, rgba(31, 105, 170, .22), transparent 32%),
          radial-gradient(circle at 92% 12%, rgba(28, 205, 226, .10), transparent 28%),
          linear-gradient(145deg, #06101d 0%, #091729 52%, #07111f 100%);
        color: var(--text);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1728, #0b2034);
        border-right: 1px solid rgba(66, 217, 255, .18);
    }
    h1, h2, h3 { color: #eaf7ff !important; }
    p, label, .stMarkdown { color: #d6e6f8; }
    .hero {
        padding: 1.35rem 1.5rem;
        border: 1px solid rgba(66,217,255,.30);
        border-radius: 22px;
        background: linear-gradient(120deg, rgba(16,36,58,.96), rgba(8,24,42,.82));
        box-shadow: 0 12px 38px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.04);
        animation: fadeUp .55s ease-out both;
        margin-bottom: 1rem;
    }
    .hero-kicker {
        color: #42d9ff;
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: .76rem;
        font-weight: 800;
    }
    .hero-title {
        font-size: clamp(1.8rem, 3vw, 2.8rem);
        font-weight: 850;
        margin: .2rem 0 .35rem 0;
        color: #f2f9ff;
    }
    .hero-sub { color: #a9c0d8; font-size: 1rem; }
    .metric-card {
        border: 1px solid rgba(90,178,230,.22);
        background: linear-gradient(145deg, rgba(16,36,58,.9), rgba(10,25,43,.9));
        padding: 1rem 1.1rem;
        border-radius: 16px;
        min-height: 95px;
        animation: fadeUp .6s ease-out both;
    }
    .metric-label { color: #9eb4cc; font-size: .82rem; }
    .metric-value { color: #eaf8ff; font-size: 1.45rem; font-weight: 800; margin-top: .2rem; }
    div.stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(66,217,255,.48);
        background: linear-gradient(100deg, #1166a4, #087f9c);
        color: white;
        font-weight: 750;
        transition: all .22s ease;
        box-shadow: 0 4px 18px rgba(10,140,190,.12);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        border-color: #8cecff;
        box-shadow: 0 8px 25px rgba(40,200,240,.22);
        background: linear-gradient(100deg, #167cc0, #089bb7);
        color: white;
    }
    div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
        border: 1px solid rgba(66,217,255,.18);
        border-radius: 12px;
        overflow: hidden;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(66,217,255,.18);
        border-radius: 12px;
        background: rgba(11,28,47,.45);
    }
    .small-note { color: #9eb4cc; font-size: .85rem; }
    .success-panel {
        border: 1px solid rgba(65, 220, 170, .35);
        background: rgba(19, 95, 83, .16);
        border-radius: 14px;
        padding: .9rem 1rem;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(9px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation: none !important; transition: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# UTILITAS MATRIKS
# =========================================================
def blank_df(rows: int, cols: int, fill: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        np.full((rows, cols), fill, dtype=float),
        index=[f"Baris {i+1}" for i in range(rows)],
        columns=[f"Kolom {j+1}" for j in range(cols)],
    )


def read_editor(df: pd.DataFrame, name: str) -> np.ndarray:
    """Convert editor values into a finite numeric NumPy array."""
    try:
        arr = df.to_numpy(dtype=float)
    except (ValueError, TypeError):
        raise ValueError(f"{name} harus berisi angka saja.")
    if arr.size == 0:
        raise ValueError(f"{name} tidak boleh kosong.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} mengandung nilai kosong atau bukan angka.")
    return arr


def format_matrix(arr: np.ndarray, precision: int = 6) -> pd.DataFrame:
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return pd.DataFrame(arr)
    rounded = np.round(arr.astype(float), precision)
    return pd.DataFrame(rounded)


def show_matrix(label: str, arr: np.ndarray, precision: int = 6):
    st.markdown(f"**{label}**")
    st.dataframe(
        format_matrix(arr, precision),
        use_container_width=True,
        height=min(440, 38 + 35 * min(np.asarray(arr).shape[0], 10)),
    )


def fmt_num(x, digits=6):
    try:
        x = float(x)
        if abs(x) < 10 ** (-digits):
            x = 0.0
        return f"{x:.{digits}g}"
    except (TypeError, ValueError):
        return str(x)


def lu_doolittle(A: np.ndarray, tol: float = 1e-12):
    """A = L U; diagonal L is 1. No row pivoting."""
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("Doolittle memerlukan matriks persegi.")
    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)
    steps = []
    for i in range(n):
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(i))
        if abs(U[i, i]) < tol:
            raise ValueError(
                f"Pivot U[{i+1},{i+1}] nol atau terlalu kecil. "
                "Metode Doolittle tanpa pivoting tidak dapat dilanjutkan untuk matriks ini."
            )
        for k in range(i + 1, n):
            L[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(i))) / U[i, i]
        if n <= 12:
            steps.append(
                f"Kolom/baris ke-{i+1}: hitung elemen U pada baris {i+1}, "
                f"lalu elemen L di bawah diagonal dengan pembagi U[{i+1},{i+1}] = {fmt_num(U[i,i])}."
            )
    return L, U, steps


def lu_crout(A: np.ndarray, tol: float = 1e-12):
    """A = L U; diagonal U is 1. No row pivoting."""
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("Crout memerlukan matriks persegi.")
    L = np.zeros((n, n), dtype=float)
    U = np.eye(n, dtype=float)
    steps = []
    for j in range(n):
        for i in range(j, n):
            L[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(j))
        if abs(L[j, j]) < tol:
            raise ValueError(
                f"Pivot L[{j+1},{j+1}] nol atau terlalu kecil. "
                "Metode Crout tanpa pivoting tidak dapat dilanjutkan untuk matriks ini."
            )
        for i in range(j + 1, n):
            U[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(j))) / L[j, j]
        if n <= 12:
            steps.append(
                f"Kolom ke-{j+1}: hitung elemen L pada kolom {j+1}, "
                f"kemudian elemen U di sebelah kanan diagonal dengan pembagi L[{j+1},{j+1}] = {fmt_num(L[j,j])}."
            )
    return L, U, steps


def cholesky_factor(A: np.ndarray, tol: float = 1e-10):
    """A = L L^T for symmetric positive-definite A."""
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("Cholesky memerlukan matriks persegi.")
    if not np.allclose(A, A.T, atol=tol, rtol=tol):
        raise ValueError("Matriks tidak simetris. Cholesky memerlukan A = Aᵀ.")
    L = np.zeros((n, n), dtype=float)
    steps = []
    for i in range(n):
        diag = A[i, i] - np.dot(L[i, :i], L[i, :i])
        if diag <= tol:
            raise ValueError(
                "Matriks tidak definit positif (atau pivot terlalu kecil). "
                "Cholesky real tidak dapat digunakan."
            )
        L[i, i] = np.sqrt(diag)
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], L[i, :i])) / L[i, i]
        if n <= 12:
            steps.append(
                f"L[{i+1},{i+1}] = √({fmt_num(A[i,i])} − jumlah kuadrat elemen sebelumnya) "
                f"= {fmt_num(L[i,i])}; lalu hitung elemen di bawahnya."
            )
    return L, steps


def rref_with_steps(A: np.ndarray, tol: float = 1e-10):
    """Reduced row echelon form, with concise elementary-operation log."""
    M = np.array(A, dtype=float).copy()
    rows, cols = M.shape
    pivot_row = 0
    steps = []
    for col in range(cols):
        if pivot_row >= rows:
            break
        candidate = pivot_row + int(np.argmax(np.abs(M[pivot_row:, col])))
        if abs(M[candidate, col]) <= tol:
            continue
        if candidate != pivot_row:
            M[[pivot_row, candidate]] = M[[candidate, pivot_row]]
            steps.append(f"Tukar baris R{pivot_row+1} ↔ R{candidate+1}.")
        pivot = M[pivot_row, col]
        M[pivot_row] /= pivot
        steps.append(f"Bagi baris R{pivot_row+1} dengan {fmt_num(pivot)} agar pivot menjadi 1.")
        for r in range(rows):
            if r != pivot_row:
                factor = M[r, col]
                if abs(factor) > tol:
                    M[r] -= factor * M[pivot_row]
                    steps.append(
                        f"R{r+1} ← R{r+1} − ({fmt_num(factor)}) R{pivot_row+1}."
                    )
        pivot_row += 1
    M[np.abs(M) < tol] = 0.0
    return M, steps, pivot_row


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Numerical computing workspace</div>
      <div class="hero-title">🧮 MatrixLab</div>
      <div class="hero-sub">Kalkulator matriks modern untuk operasi aljabar linear, faktorisasi, dan penyelesaian sistem persamaan linear.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## ⚙️ Pengaturan")
st.sidebar.caption("Pilih operasi dan ukuran matriks.")
operation = st.sidebar.selectbox(
    "Pilih operasi",
    [
        "Penjumlahan",
        "Pengurangan",
        "Perkalian",
        "Transpose",
        "Determinan",
        "Invers (Matriks Identitas)",
        "Rank",
        "Trace",
        "SPL - Gauss-Jordan",
        "Doolittle Factorization",
        "Crout Factorization",
        "Cholesky Factorization",
    ],
)
need_b = operation in ("Penjumlahan", "Pengurangan", "Perkalian", "SPL - Gauss-Jordan")

st.sidebar.markdown("---")
st.sidebar.markdown("### Ukuran Matriks A")
a_rows = st.sidebar.number_input("Baris A", min_value=1, max_value=100, value=3, step=1)
a_cols = st.sidebar.number_input("Kolom A", min_value=1, max_value=100, value=3, step=1)

if need_b:
    st.sidebar.markdown("### Ukuran Matriks B")
    if operation == "Perkalian":
        default_b_rows = int(a_cols)
    elif operation == "SPL - Gauss-Jordan":
        default_b_rows = int(a_rows)
    else:
        default_b_rows = int(a_rows)
    b_rows = st.sidebar.number_input("Baris B", min_value=1, max_value=100, value=default_b_rows, step=1)
    b_cols = st.sidebar.number_input("Kolom B", min_value=1, max_value=100, value=1 if operation == "SPL - Gauss-Jordan" else int(a_cols), step=1)
else:
    b_rows, b_cols = 1, 1

precision = st.sidebar.select_slider("Angka desimal hasil", options=[2, 4, 6, 8, 10], value=6)
show_steps = st.sidebar.checkbox("Tampilkan langkah perhitungan", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("Mendukung ukuran hingga 100 × 100. Untuk matriks besar, tampilan langkah dibatasi agar aplikasi tetap responsif.")

# =========================================================
# INPUT MATRIX
# =========================================================
col_a, col_b = st.columns(2 if need_b else 1, gap="large")

with col_a:
    st.subheader("Matriks A")
    st.caption(f"{int(a_rows)} baris × {int(a_cols)} kolom")
    key_a = f"editor_A_{operation}_{a_rows}_{a_cols}"
    default_a = blank_df(int(a_rows), int(a_cols))
    # Small educational example for the initial 3x3 matrix
    if int(a_rows) == 3 and int(a_cols) == 3 and key_a not in st.session_state:
        default_a.iloc[:, :] = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]], dtype=float)
    df_a = st.data_editor(
        default_a,
        key=key_a,
        num_rows="fixed",
        use_container_width=True,
        height=min(460, 40 + 35 * min(int(a_rows), 11)),
        hide_index=False,
    )

if need_b:
    with col_b:
        st.subheader("Matriks B")
        st.caption(f"{int(b_rows)} baris × {int(b_cols)} kolom")
        key_b = f"editor_B_{operation}_{b_rows}_{b_cols}"
        default_b = blank_df(int(b_rows), int(b_cols))
        if operation == "SPL - Gauss-Jordan":
            default_b.iloc[:, 0] = 1.0
        df_b = st.data_editor(
            default_b,
            key=key_b,
            num_rows="fixed",
            use_container_width=True,
            height=min(460, 40 + 35 * min(int(b_rows), 11)),
            hide_index=False,
        )
else:
    df_b = None

st.markdown("")
run = st.button("▶  Hitung Matriks", type="primary", use_container_width=True)

if run:
    try:
        A = read_editor(df_a, "Matriks A")
        B = read_editor(df_b, "Matriks B") if df_b is not None else None

        st.markdown("---")
        st.header("Hasil Perhitungan")

        if operation == "Penjumlahan":
            if A.shape != B.shape:
                raise ValueError("Penjumlahan memerlukan ukuran A dan B yang sama.")
            result = A + B
            show_matrix("A + B", result, precision)
            st.success("Penjumlahan berhasil.")

        elif operation == "Pengurangan":
            if A.shape != B.shape:
                raise ValueError("Pengurangan memerlukan ukuran A dan B yang sama.")
            result = A - B
            show_matrix("A − B", result, precision)
            st.success("Pengurangan berhasil.")

        elif operation == "Perkalian":
            if A.shape[1] != B.shape[0]:
                raise ValueError(
                    f"Kolom A ({A.shape[1]}) harus sama dengan baris B ({B.shape[0]})."
                )
            result = A @ B
            show_matrix("A × B", result, precision)
            st.success("Perkalian berhasil.")
            if show_steps and A.shape[0] <= 8 and A.shape[1] <= 8 and B.shape[1] <= 8:
                with st.expander("Langkah perkalian"):
                    st.write("Setiap elemen hasil diperoleh dari perkalian titik baris A dengan kolom B.")
                    st.latex(r"C_{ij}=\sum_{k=1}^{n} A_{ik}B_{kj}")
                    st.write(f"Ukuran hasil: {result.shape[0]} × {result.shape[1]}.")

        elif operation == "Transpose":
            result = A.T
            show_matrix("Transpose A (Aᵀ)", result, precision)
            st.success("Transpose berhasil.")
            if show_steps:
                with st.expander("Penjelasan"):
                    st.write("Baris pada A menjadi kolom pada Aᵀ, dan kolom pada A menjadi baris.")

        elif operation == "Determinan":
            if A.shape[0] != A.shape[1]:
                raise ValueError("Determinan hanya dapat dihitung untuk matriks persegi.")
            det = float(np.linalg.det(A))
            st.metric("det(A)", fmt_num(det, precision))
            if show_steps:
                with st.expander("Penjelasan"):
                    st.write("Determinan dihitung secara numerik menggunakan eliminasi/faktorisasi linear.")
                    if A.shape[0] <= 8:
                        st.write("Matriks A:")
                        st.dataframe(format_matrix(A, precision), use_container_width=True)

        elif operation == "Invers (Matriks Identitas)":
            if A.shape[0] != A.shape[1]:
                raise ValueError("Invers hanya dapat dihitung untuk matriks persegi.")
            det = float(np.linalg.det(A))
            if abs(det) < 1e-12:
                raise ValueError("Matriks singular (determinan nol atau sangat kecil), sehingga tidak memiliki invers.")
            inv = np.linalg.inv(A)
            show_matrix("A⁻¹", inv, precision)
            st.success("Invers berhasil dihitung.")
            if show_steps:
                with st.expander("Metode matriks identitas: [A | I] → [I | A⁻¹]"):
                    st.latex(r"[A \mid I] \longrightarrow [I \mid A^{-1}]")
                    st.write("1. Gabungkan A dengan matriks identitas I berukuran sama.")
                    st.write("2. Lakukan operasi baris elementer hingga bagian kiri menjadi I.")
                    st.write("3. Bagian kanan menjadi invers A⁻¹.")
                    if A.shape[0] <= 8:
                        augmented = np.hstack([A, np.eye(A.shape[0])])
                        show_matrix("[A | I] awal", augmented, precision)
                        reduced, log, _ = rref_with_steps(augmented)
                        show_matrix("[I | A⁻¹] akhir", reduced, precision)
                        for step in log:
                            st.write("• " + step)
                    else:
                        st.info("Untuk menjaga kecepatan, matriks gabungan dan rincian operasi baris ditampilkan hanya untuk ukuran sampai 8 × 8.")

        elif operation == "Rank":
            reduced, log, rank = rref_with_steps(A)
            st.metric("Rank(A)", rank)
            show_matrix("Bentuk eselon baris tereduksi (RREF)", reduced, precision)
            if show_steps:
                with st.expander("Langkah operasi baris"):
                    if len(log) > 100:
                        st.info("Log sangat panjang; ditampilkan 100 langkah pertama.")
                    for step in log[:100]:
                        st.write("• " + step)

        elif operation == "Trace":
            if A.shape[0] != A.shape[1]:
                raise ValueError("Trace hanya dapat dihitung untuk matriks persegi.")
            trace = float(np.trace(A))
            st.metric("tr(A)", fmt_num(trace, precision))
            if show_steps:
                with st.expander("Penjelasan"):
                    st.latex(r"\operatorname{tr}(A)=\sum_{i=1}^{n} a_{ii}")
                    st.write("Trace adalah jumlah elemen pada diagonal utama.")

        elif operation == "SPL - Gauss-Jordan":
            if A.shape[0] != A.shape[1]:
                raise ValueError("Untuk SPL ini, matriks koefisien A harus persegi.")
            if B.shape[0] != A.shape[0] or B.shape[1] != 1:
                raise ValueError("Matriks B untuk SPL harus berupa satu kolom dengan jumlah baris yang sama dengan A.")
            augmented = np.hstack([A, B])
            reduced, log, _ = rref_with_steps(augmented)
            show_matrix("Matriks augmented [A | B]", augmented, precision)
            show_matrix("Hasil Gauss-Jordan", reduced, precision)
            n = A.shape[0]
            left = reduced[:, :n]
            right = reduced[:, n:]
            if np.linalg.matrix_rank(A) == np.linalg.matrix_rank(augmented):
                if np.linalg.matrix_rank(A) == n:
                    st.success("SPL memiliki solusi tunggal.")
                    for i, value in enumerate(right[:, 0], start=1):
                        st.write(f"x{i} = **{fmt_num(value, precision)}**")
                else:
                    st.warning("SPL memiliki banyak solusi atau variabel bebas. Periksa bentuk RREF untuk melihat hubungan variabel.")
            else:
                st.error("SPL tidak konsisten: tidak ada solusi.")
            if show_steps:
                with st.expander("Langkah operasi baris Gauss-Jordan"):
                    if len(log) > 100:
                        st.info("Log sangat panjang; ditampilkan 100 langkah pertama.")
                    for step in log[:100]:
                        st.write("• " + step)

        elif operation == "Doolittle Factorization":
            L, U, steps = lu_doolittle(A)
            show_matrix("Matriks L (diagonal utama = 1)", L, precision)
            show_matrix("Matriks U", U, precision)
            err = np.linalg.norm(A - L @ U)
            st.success(f"Faktorisasi selesai. Galat rekonstruksi ||A − LU|| = {err:.3e}")
            if show_steps:
                with st.expander("Rumus dan langkah Doolittle"):
                    st.latex(r"A=LU,\quad l_{ii}=1")
                    st.write("Rumus elemen U:")
                    st.latex(r"u_{ij}=a_{ij}-\sum_{k=1}^{i-1}l_{ik}u_{kj}")
                    st.write("Rumus elemen L untuk i > j:")
                    st.latex(r"l_{ij}=\frac{a_{ij}-\sum_{k=1}^{j-1}l_{ik}u_{kj}}{u_{jj}}")
                    for step in steps:
                        st.write("• " + step)
                    if A.shape[0] > 12:
                        st.info("Untuk matriks di atas 12 × 12, rincian langkah per elemen diringkas agar aplikasi tetap responsif.")

        elif operation == "Crout Factorization":
            L, U, steps = lu_crout(A)
            show_matrix("Matriks L", L, precision)
            show_matrix("Matriks U (diagonal utama = 1)", U, precision)
            err = np.linalg.norm(A - L @ U)
            st.success(f"Faktorisasi selesai. Galat rekonstruksi ||A − LU|| = {err:.3e}")
            if show_steps:
                with st.expander("Rumus dan langkah Crout"):
                    st.latex(r"A=LU,\quad u_{ii}=1")
                    st.write("Rumus elemen L:")
                    st.latex(r"l_{ij}=a_{ij}-\sum_{k=1}^{j-1}l_{ik}u_{kj}")
                    st.write("Rumus elemen U untuk j > i:")
                    st.latex(r"u_{ij}=\frac{a_{ij}-\sum_{k=1}^{i-1}l_{ik}u_{kj}}{l_{ii}}")
                    for step in steps:
                        st.write("• " + step)
                    if A.shape[0] > 12:
                        st.info("Untuk matriks di atas 12 × 12, rincian langkah per elemen diringkas agar aplikasi tetap responsif.")

        elif operation == "Cholesky Factorization":
            L, steps = cholesky_factor(A)
            show_matrix("Matriks L", L, precision)
            show_matrix("Transpose L (Lᵀ)", L.T, precision)
            err = np.linalg.norm(A - L @ L.T)
            st.success(f"Faktorisasi selesai. Galat rekonstruksi ||A − LLᵀ|| = {err:.3e}")
            if show_steps:
                with st.expander("Rumus dan langkah Cholesky"):
                    st.latex(r"A=LL^T")
                    st.latex(r"l_{ii}=\sqrt{a_{ii}-\sum_{k=1}^{i-1}l_{ik}^{2}}")
                    st.latex(r"l_{ji}=\frac{a_{ji}-\sum_{k=1}^{i-1}l_{jk}l_{ik}}{l_{ii}},\quad j>i")
                    st.info("Syarat: A harus simetris dan definit positif.")
                    for step in steps:
                        st.write("• " + step)
                    if A.shape[0] > 12:
                        st.info("Untuk matriks di atas 12 × 12, rincian langkah per elemen diringkas agar aplikasi tetap responsif.")

    except np.linalg.LinAlgError as e:
        st.error(f"Perhitungan gagal: {e}. Periksa ukuran dan nilai matriks.")
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Terjadi kesalahan yang tidak terduga: {e}")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8fa9c4;font-size:.82rem;padding:.4rem'>"
    "MatrixLab · Python + Streamlit · Untuk pembelajaran aljabar linear"
    "</div>",
    unsafe_allow_html=True,
)
'''
req = "streamlit\nnumpy\npandas\nsympy\n"

base = Path("/mnt/data")
base.mkdir(parents=True, exist_ok=True)
(base / "app.py").write_text(app_code, encoding="utf-8")
(base / "requirements.txt").write_text(req, encoding="utf-8")
print("Berhasil membuat app.py dan requirements.txt")
