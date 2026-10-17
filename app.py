import streamlit as st
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
    """A = L U; diagonal L is 1. Logs numeric substitution for each element."""
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("Doolittle memerlukan matriks persegi.")
    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)
    steps = []
    for i in range(n):
        for j in range(i, n):
            terms = [f"L[{i+1},{k+1}]×U[{k+1},{j+1}]" for k in range(i)]
            total = sum(L[i, k] * U[k, j] for k in range(i))
            formula = " + ".join(terms) if terms else "0"
            U[i, j] = A[i, j] - total
            if n <= 12:
                values = " + ".join(fmt_num(L[i,k]) + "×" + fmt_num(U[k,j]) for k in range(i)) or "0"
                steps.append(f"**Hitung U[{i+1},{j+1}]**: U[{i+1},{j+1}] = A[{i+1},{j+1}] − Σ(L×U) = {fmt_num(A[i,j])} − ({values}) = **{fmt_num(U[i,j])}**.")
        if abs(U[i, i]) < tol:
            raise ValueError(f"Pivot U[{i+1},{i+1}] nol atau terlalu kecil. Metode Doolittle tanpa pivoting tidak dapat dilanjutkan untuk matriks ini.")
        for j in range(i + 1, n):
            total = sum(L[j, k] * U[k, i] for k in range(i))
            values = " + ".join(fmt_num(L[j,k]) + "×" + fmt_num(U[k,i]) for k in range(i)) or "0"
            L[j, i] = (A[j, i] - total) / U[i, i]
            if n <= 12:
                steps.append(f"**Hitung L[{j+1},{i+1}]**: L[{j+1},{i+1}] = (A[{j+1},{i+1}] − Σ(L×U)) / U[{i+1},{i+1}] = ({fmt_num(A[j,i])} − ({values})) / {fmt_num(U[i,i])} = **{fmt_num(L[j,i])}**.")
    return L, U, steps


def lu_crout(A: np.ndarray, tol: float = 1e-12):
    """A = L U; diagonal U is 1. Logs numeric substitution for each element."""
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("Crout memerlukan matriks persegi.")
    L = np.zeros((n, n), dtype=float)
    U = np.eye(n, dtype=float)
    steps = []
    for j in range(n):
        for i in range(j, n):
            total = sum(L[i, k] * U[k, j] for k in range(j))
            values = " + ".join(fmt_num(L[i,k]) + "×" + fmt_num(U[k,j]) for k in range(j)) or "0"
            L[i, j] = A[i, j] - total
            if n <= 12:
                steps.append(f"**Hitung L[{i+1},{j+1}]**: L[{i+1},{j+1}] = A[{i+1},{j+1}] − Σ(L×U) = {fmt_num(A[i,j])} − ({values}) = **{fmt_num(L[i,j])}**.")
        if abs(L[j, j]) < tol:
            raise ValueError(f"Pivot L[{j+1},{j+1}] nol atau terlalu kecil. Metode Crout tanpa pivoting tidak dapat dilanjutkan untuk matriks ini.")
        for i in range(j + 1, n):
            total = sum(L[j, k] * U[k, i] for k in range(j))
            values = " + ".join(fmt_num(L[j,k]) + "×" + fmt_num(U[k,i]) for k in range(j)) or "0"
            U[j, i] = (A[j, i] - total) / L[j, j]
            if n <= 12:
                steps.append(f"**Hitung U[{j+1},{i+1}]**: U[{j+1},{i+1}] = (A[{j+1},{i+1}] − Σ(L×U)) / L[{j+1},{j+1}] = ({fmt_num(A[j,i])} − ({values})) / {fmt_num(L[j,j])} = **{fmt_num(U[j,i])}**.")
    return L, U, steps


def symbolic_cholesky_product(n: int) -> str:
    """Return LaTeX showing symbolic L multiplied by its transpose."""
    if n > 5:
        return ""
    symbols = {}
    for i in range(n):
        for j in range(i + 1):
            symbols[(i, j)] = sp.Symbol(f"l_{{{i+1}{j+1}}}")
    Lsym = sp.zeros(n, n)
    for (i, j), symbol in symbols.items():
        Lsym[i, j] = symbol
    product = sp.simplify(Lsym * Lsym.T)
    return sp.latex(Lsym) + r"\;\times\;" + sp.latex(Lsym.T) + r"\;=\;" + sp.latex(product)


def cholesky_factor(A: np.ndarray, tol: float = 1e-10):
    """Hitung A = L L^T dan buat langkah manual yang rapi untuk setiap elemen L."""
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("Cholesky memerlukan matriks persegi.")
    if not np.allclose(A, A.T, atol=tol, rtol=tol):
        raise ValueError("Matriks tidak simetris. Cholesky memerlukan A = Aᵀ.")

    L = np.zeros((n, n), dtype=float)
    steps = []

    # Urutan pengerjaan manual: l11, l21, l31, ..., l22, l32, ..., l33.
    for i in range(n):
        # 1) Elemen diagonal l_ii:
        # a_ii = l_i1^2 + ... + l_i,i-1^2 + l_ii^2
        known_values = [float(L[i, k]) ** 2 for k in range(i)]
        known_sum = float(sum(known_values))
        residual = float(A[i, i] - known_sum)
        if residual <= tol:
            raise ValueError(
                f"Matriks tidak definit positif: nilai di bawah akar untuk l{i+1}{i+1} "
                f"adalah {fmt_num(residual)}, sehingga Cholesky real tidak dapat dilanjutkan."
            )
        L[i, i] = np.sqrt(residual)
        label = f"l_{{{i+1}{i+1}}}"

        if i == 0:
            equation = f"a_{{11}} = l_{{11}}^2"
            substitution = f"{fmt_num(A[i, i])} = l_{{11}}^2"
            simplify = f"l_{{11}} = \\sqrt{{{fmt_num(A[i, i])}}} = \\mathbf{{{fmt_num(L[i, i])}}}"
        else:
            terms = [f"l_{{{i+1}{k+1}}}^2" for k in range(i)]
            equation = f"a_{{{i+1}{i+1}}} = " + " + ".join(terms + [f"l_{{{i+1}{i+1}}}^2"])
            known_expr = " + ".join(f"({fmt_num(L[i, k])})^2" for k in range(i))
            substitution = f"{fmt_num(A[i, i])} = {known_expr} + l_{{{i+1}{i+1}}}^2"
            simplify = (
                f"l_{{{i+1}{i+1}}} = \\sqrt{{{fmt_num(A[i, i])} - ({fmt_num(known_sum)})}}"
                f" = \\sqrt{{{fmt_num(residual)}}} = \\mathbf{{{fmt_num(L[i, i])}}}"
            )

        steps.append(
            f"**Cari \\({label}\\) dari elemen \\(A[{i+1},{i+1}]\\)**\n\n"
            f"**Rumus dari perkalian \\(LL^T\\):**\n\n"
            f"\\[{equation}\\]\n\n"
            f"**Substitusi angka:**\n\n"
            f"\\[{substitution}\\]\n\n"
            f"**Selesaikan:**\n\n"
            f"\\[{simplify}\\]"
        )

        # 2) Elemen di bawah diagonal pada kolom i:
        # a_ji = l_j1*l_i1 + ... + l_ji*l_ii, j > i
        for j in range(i + 1, n):
            prior_terms_symbolic = [f"l_{{{j+1}{k+1}}}l_{{{i+1}{k+1}}}" for k in range(i)]
            equation_terms = prior_terms_symbolic + [f"l_{{{j+1}{i+1}}}l_{{{i+1}{i+1}}}"]
            prior_total = float(np.dot(L[j, :i], L[i, :i]))
            L[j, i] = (float(A[j, i]) - prior_total) / L[i, i]

            numeric_terms = [f"({fmt_num(L[j, k])})({fmt_num(L[i, k])})" for k in range(i)]
            numeric_terms.append(f"l_{{{j+1}{i+1}}}({fmt_num(L[i, i])})")
            symbolic_equation = " + ".join(equation_terms)
            numeric_equation = " + ".join(numeric_terms)
            result_expr = (
                f"l_{{{j+1}{i+1}}} = \\frac{{{fmt_num(A[j, i])} - ({fmt_num(prior_total)})}}"
                f"{{{fmt_num(L[i, i])}}} = \\mathbf{{{fmt_num(L[j, i])}}}"
            )
            steps.append(
                f"**Cari \\(l_{{{j+1}{i+1}}}\\) dari elemen \\(A[{j+1},{i+1}]\\)**\n\n"
                f"**Rumus dari perkalian \\(LL^T\\):**\n\n"
                f"\\[a_{{{j+1}{i+1}}} = {symbolic_equation}\\]\n\n"
                f"**Substitusi angka:**\n\n"
                f"\\[{fmt_num(A[j, i])} = {numeric_equation}\\]\n\n"
                f"**Pindahkan suku yang sudah diketahui, lalu bagi dengan \\(l_{{{i+1}{i+1}}}\\):**\n\n"
                f"\\[{result_expr}\\]"
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



def rref_with_snapshots(A: np.ndarray, tol: float = 1e-10):
    """RREF with a copy of the augmented/matrix state after every row operation."""
    M = np.array(A, dtype=float).copy()
    rows, cols = M.shape
    pivot_row = 0
    steps = []

    def record(description):
        snapshot = M.copy()
        snapshot[np.abs(snapshot) < tol] = 0.0
        steps.append((description, snapshot))

    for col in range(cols):
        if pivot_row >= rows:
            break
        candidate = pivot_row + int(np.argmax(np.abs(M[pivot_row:, col])))
        if abs(M[candidate, col]) <= tol:
            continue
        if candidate != pivot_row:
            M[[pivot_row, candidate]] = M[[candidate, pivot_row]]
            record(f"Tukar baris R{pivot_row+1} ↔ R{candidate+1}.")
        pivot = M[pivot_row, col]
        M[pivot_row] /= pivot
        record(f"Bagi seluruh baris R{pivot_row+1} dengan {fmt_num(pivot)} supaya pivot menjadi 1.")
        for r in range(rows):
            if r != pivot_row:
                factor = M[r, col]
                if abs(factor) > tol:
                    M[r] -= factor * M[pivot_row]
                    record(f"Hilangkan elemen kolom {col+1} pada R{r+1}: R{r+1} ← R{r+1} − ({fmt_num(factor)})R{pivot_row+1}.")
        pivot_row += 1
    M[np.abs(M) < tol] = 0.0
    return M, steps, pivot_row


def show_row_operation_steps(title, steps, precision=6, limit=60):
    """Render row operations and the matrix after each operation for learning."""
    st.markdown(f"**{title}**")
    if not steps:
        st.info("Matriks sudah berada dalam bentuk yang diperlukan; tidak ada operasi baris tambahan.")
        return
    if len(steps) > limit:
        st.info(f"Ada {len(steps)} operasi. Demi menjaga aplikasi tetap responsif, ditampilkan {limit} operasi pertama beserta matriks setelah operasi.")
    for idx, (description, snapshot) in enumerate(steps[:limit], start=1):
        with st.expander(f"Langkah {idx}: {description}", expanded=(idx == 1)):
            st.dataframe(format_matrix(snapshot, precision), use_container_width=True)

# =========================================================
# MODUL PEMBELAJARAN
# =========================================================
LESSONS = {
    "Penjumlahan": {
        "title": "Penjumlahan Matriks",
        "goal": "Menjumlahkan elemen yang posisinya sama pada dua matriks.",
        "idea": "Ukuran A dan B harus sama. Elemen baris ke-i kolom ke-j dijumlahkan dengan elemen pada posisi yang sama.",
        "formula": r"C=A+B,\quad c_{ij}=a_{ij}+b_{ij}",
        "example": "Jika A[1,1] = 2 dan B[1,1] = 5, maka C[1,1] = 2 + 5 = 7.",
        "check": "Pastikan jumlah baris dan kolom kedua matriks sama."
    },
    "Pengurangan": {
        "title": "Pengurangan Matriks",
        "goal": "Mengurangkan elemen B dari elemen A pada posisi yang sama.",
        "idea": "Ukuran A dan B harus sama. Urutan penting: A − B tidak selalu sama dengan B − A.",
        "formula": r"C=A-B,\quad c_{ij}=a_{ij}-b_{ij}",
        "example": "Jika A[1,1] = 8 dan B[1,1] = 3, maka C[1,1] = 8 − 3 = 5.",
        "check": "Jangan menukar urutan pengurangan."
    },
    "Perkalian": {
        "title": "Perkalian Matriks",
        "goal": "Mengalikan baris matriks A dengan kolom matriks B.",
        "idea": "Jumlah kolom A harus sama dengan jumlah baris B. Untuk satu elemen hasil, kalikan pasangan angka lalu jumlahkan hasil perkaliannya.",
        "formula": r"c_{ij}=\sum_{k=1}^{n}a_{ik}b_{kj}",
        "example": "Baris [2, 3] dikali kolom [4, 5] menghasilkan (2 × 4) + (3 × 5) = 23.",
        "check": "Ukuran hasil adalah (baris A) × (kolom B)."
    },
    "Transpose": {
        "title": "Transpose Matriks",
        "goal": "Mengubah setiap baris menjadi kolom.",
        "idea": "Elemen pada posisi baris i dan kolom j berpindah ke baris j dan kolom i.",
        "formula": r"(A^T)_{ij}=A_{ji}",
        "example": "Jika A = [[1, 2], [3, 4]], maka Aᵀ = [[1, 3], [2, 4]].",
        "check": "Jika A berukuran m × n, maka Aᵀ berukuran n × m."
    },
    "Determinan": {
        "title": "Determinan Matriks",
        "goal": "Menghitung satu nilai skalar dari matriks persegi.",
        "idea": "Determinan hanya didefinisikan untuk matriks persegi. Untuk 2 × 2, kalikan diagonal utama lalu kurangi dengan perkalian diagonal lainnya.",
        "formula": r"A=\begin{bmatrix}a&b\c&d\end{bmatrix},\quad \det(A)=ad-bc",
        "example": "Untuk [[2, 3], [1, 4]], det(A) = (2 × 4) − (3 × 1) = 5.",
        "check": "Jika determinan nol, matriks tidak memiliki invers."
    },
    "Invers (Matriks Identitas)": {
        "title": "Invers dengan Matriks Identitas",
        "goal": "Mencari matriks A⁻¹ yang jika dikalikan dengan A menghasilkan matriks identitas.",
        "idea": "Gabungkan A dan I menjadi [A | I]. Gunakan operasi baris elementer sampai bagian kiri menjadi I; bagian kanan akan menjadi A⁻¹.",
        "formula": r"[A\mid I]\longrightarrow[I\mid A^{-1}],\quad AA^{-1}=I",
        "example": "Matriks identitas 2 × 2 adalah [[1, 0], [0, 1]]. Elemen diagonalnya 1, elemen lainnya 0.",
        "check": "A harus persegi dan determinannya tidak boleh nol."
    },
    "Rank": {
        "title": "Rank Matriks",
        "goal": "Menentukan jumlah pivot atau baris tak nol pada bentuk eselon baris.",
        "idea": "Sederhanakan matriks dengan operasi baris elementer. Jumlah pivot pada bentuk eselon adalah rank matriks.",
        "formula": r"\operatorname{rank}(A)=\text{jumlah pivot}",
        "example": "Jika setelah eliminasi terdapat dua baris tak nol dan satu baris nol, rank matriks adalah 2.",
        "check": "Rank tidak mungkin melebihi jumlah baris atau kolom."
    },
    "Trace": {
        "title": "Trace Matriks",
        "goal": "Menjumlahkan elemen pada diagonal utama matriks persegi.",
        "idea": "Ambil elemen dari kiri atas ke kanan bawah, lalu jumlahkan.",
        "formula": r"\operatorname{tr}(A)=a_{11}+a_{22}+\cdots+a_{nn}",
        "example": "Untuk [[2, 9], [4, 3]], trace = 2 + 3 = 5.",
        "check": "Trace hanya dihitung untuk matriks persegi."
    },
    "SPL - Gauss-Jordan": {
        "title": "SPL dengan Gauss–Jordan",
        "goal": "Mencari nilai variabel dengan mengubah matriks augmented menjadi bentuk eselon baris tereduksi.",
        "idea": "Bentuk [A | B]. Gunakan pertukaran baris, perkalian baris dengan bilangan bukan nol, atau penjumlahan kelipatan baris untuk membentuk pivot 1 dan nol di atas/bawah pivot.",
        "formula": r"[A\mid B]\longrightarrow[I\mid x]",
        "example": "Dari x + y = 3 dan x − y = 1, jumlahkan kedua persamaan sehingga 2x = 4, jadi x = 2; kemudian y = 1.",
        "check": "Substitusikan nilai variabel ke persamaan awal untuk memeriksa hasil."
    },
    "Doolittle Factorization": {
        "title": "Faktorisasi Doolittle",
        "goal": "Memecah matriks A menjadi perkalian matriks segitiga bawah L dan segitiga atas U.",
        "idea": "Pada Doolittle, diagonal utama L bernilai 1. Elemen U dihitung terlebih dahulu, lalu elemen di bawah diagonal L dihitung menggunakan pivot U.",
        "formula": r"A=LU,\quad l_{ii}=1",
        "example": "Untuk matriks 2 × 2, bentuk L = [[1, 0], [l₂₁, 1]] dan U = [[u₁₁, u₁₂], [0, u₂₂]].",
        "check": "Metode pada aplikasi ini tidak melakukan pivoting; pivot nol dapat membuat proses gagal."
    },
    "Crout Factorization": {
        "title": "Faktorisasi Crout",
        "goal": "Memecah A menjadi L dan U seperti Doolittle, tetapi aturan diagonalnya berbeda.",
        "idea": "Pada Crout, diagonal utama U bernilai 1. Elemen L dihitung terlebih dahulu, lalu elemen U di sebelah kanan diagonal.",
        "formula": r"A=LU,\quad u_{ii}=1",
        "example": "Untuk matriks 2 × 2, bentuk L = [[l₁₁, 0], [l₂₁, l₂₂]] dan U = [[1, u₁₂], [0, 1]].",
        "check": "Matriks harus persegi dan pivot L tidak boleh nol."
    },
    "Cholesky Factorization": {
        "title": "Faktorisasi Cholesky",
        "goal": "Memecah matriks A menjadi L dikali transpose L.",
        "idea": "Cholesky hanya dapat digunakan pada matriks persegi yang simetris dan definit positif. Elemen diagonal L dihitung menggunakan akar kuadrat.",
        "formula": r"A=LL^T,\quad l_{ii}=\sqrt{a_{ii}-\sum_{k=1}^{i-1}l_{ik}^2}",
        "example": "Matriks simetris berarti A = Aᵀ, misalnya [[4, 2], [2, 3]].",
        "check": "Jika matriks tidak simetris atau bukan definit positif, Cholesky real tidak dapat dilanjutkan."
    },
}


def show_lesson(operation: str):
    lesson = LESSONS.get(operation)
    if not lesson:
        return
    st.markdown("## 📘 Belajar sebelum menghitung")
    st.markdown(f"### {lesson['title']}")
    st.markdown(f"**Tujuan:** {lesson['goal']}")
    st.info(lesson["idea"])
    st.markdown("**Rumus utama**")
    st.latex(lesson["formula"])
    with st.expander("Lihat contoh sederhana dan tips", expanded=True):
        st.write("**Contoh:** " + lesson["example"])
        st.write("**Yang perlu diingat:** " + lesson["check"])


def show_practice(operation: str):
    """Small retrieval-practice questions; reveal feedback only after submission."""
    questions = {
        "Penjumlahan": ("Jika A[1,1] = 4 dan B[1,1] = 6, berapa C[1,1] untuk C = A + B?", ["2", "10", "24"], "10", "Penjumlahan menjumlahkan elemen pada posisi yang sama: 4 + 6 = 10."),
        "Pengurangan": ("Jika A[1,1] = 9 dan B[1,1] = 2, berapa (A − B)[1,1]?", ["7", "11", "18"], "7", "Kurangi elemen B dari elemen A: 9 − 2 = 7."),
        "Perkalian": ("Baris [2, 3] dikali kolom [4, 5] menghasilkan berapa?", ["23", "26", "19"], "23", "Kalikan lalu jumlahkan: (2 × 4) + (3 × 5) = 8 + 15 = 23."),
        "Transpose": ("Pada transpose, baris matriks akan menjadi apa?", ["Baris tetap baris", "Kolom", "Diagonal"], "Kolom", "Transpose menukar posisi baris dan kolom."),
        "Determinan": ("Determinan [[2, 3], [1, 4]] adalah ...", ["5", "11", "8"], "5", "det(A) = ad − bc = (2 × 4) − (3 × 1) = 5."),
        "Invers (Matriks Identitas)": ("Jika determinan matriks persegi sama dengan nol, apakah matriks memiliki invers?", ["Ya", "Tidak", "Selalu"], "Tidak", "Determinan nol berarti matriks singular dan tidak memiliki invers."),
        "Rank": ("Rank dapat ditentukan dari jumlah ... pada bentuk eselon baris.", ["Pivot", "Kolom kosong", "Angka negatif"], "Pivot", "Rank sama dengan jumlah pivot atau baris tak nol pada bentuk eselon."),
        "Trace": ("Trace matriks persegi adalah jumlah elemen pada ...", ["Diagonal utama", "Baris pertama", "Kolom terakhir"], "Diagonal utama", "Trace menjumlahkan elemen diagonal utama dari kiri atas ke kanan bawah."),
        "SPL - Gauss-Jordan": ("Operasi baris mana yang diperbolehkan dalam Gauss–Jordan?", ["Menukar dua baris", "Mengubah hanya satu angka tanpa aturan", "Menghapus persamaan"], "Menukar dua baris", "Pertukaran baris adalah salah satu operasi baris elementer yang sah."),
        "Doolittle Factorization": ("Pada metode Doolittle, elemen diagonal utama matriks L adalah ...", ["Selalu 1", "Selalu 0", "Selalu −1"], "Selalu 1", "Doolittle menetapkan diagonal utama L bernilai 1."),
        "Crout Factorization": ("Pada metode Crout, elemen diagonal utama matriks U adalah ...", ["Selalu 1", "Selalu 0", "Selalu sama dengan diagonal A"], "Selalu 1", "Crout menetapkan diagonal utama U bernilai 1."),
        "Cholesky Factorization": ("Salah satu syarat Cholesky adalah matriks harus ...", ["Simetris dan definit positif", "Selalu berordo 2 × 3", "Memiliki semua elemen nol"], "Simetris dan definit positif", "Cholesky real membutuhkan matriks persegi simetris dan definit positif."),
    }
    q = questions.get(operation)
    if not q:
        return
    question, options, correct, explanation = q
    st.markdown("## 🧠 Latihan mandiri")
    st.write(question)
    choice = st.radio("Pilih jawaban:", options, key=f"quiz_{operation}", index=None)
    if st.button("Periksa jawaban", key=f"check_{operation}"):
        if choice is None:
            st.warning("Pilih salah satu jawaban terlebih dahulu.")
        elif choice == correct:
            st.success("Benar! " + explanation)
        else:
            st.error("Belum tepat. Coba pikirkan kembali, lalu baca petunjuk berikut.")
            st.info(explanation)

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
st.sidebar.caption("Pilih cara belajar dan operasi matriks.")
mode = st.sidebar.radio("Mode aplikasi", ["Mode Belajar", "Mode Kalkulator", "Mode Latihan"], index=0)
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
# AREA BELAJAR / LATIHAN
# =========================================================
if mode == "Mode Belajar":
    show_lesson(operation)
elif mode == "Mode Latihan":
    st.markdown("## 🎯 Uji pemahamanmu")
    st.caption("Jawab pertanyaan singkat sebelum memakai kalkulator. Penjelasan akan muncul setelah kamu memeriksa jawaban.")
    show_practice(operation)
    st.markdown("---")
    st.markdown("### Sekarang coba dengan matriksmu sendiri")
else:
    st.caption("Mode Kalkulator aktif: masukkan matriks dan hitung. Aktifkan 'Tampilkan langkah perhitungan' untuk melihat prosesnya.")

# =========================================================
# INPUT MATRIX
# =========================================================
# st.columns(1) menghasilkan satu kolom, jadi jangan unpack menjadi dua variabel.
input_columns = st.columns(2, gap="large") if need_b else st.columns(1, gap="large")
col_a = input_columns[0]
col_b = input_columns[1] if need_b else None

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
run = st.button("▶  Hitung dan Pelajari Langkahnya", type="primary", use_container_width=True)

if run:
    try:
        A = read_editor(df_a, "Matriks A")
        B = read_editor(df_b, "Matriks B") if df_b is not None else None

        st.markdown("---")
        st.header("Hasil dan Pembahasan")
        st.caption("Baca rumus dan uraian langkahnya untuk memahami bagaimana hasil diperoleh, bukan hanya menyalin angka akhir.")

        if operation == "Penjumlahan":
            if A.shape != B.shape:
                raise ValueError("Penjumlahan memerlukan ukuran A dan B yang sama.")
            result = A + B
            show_matrix("A + B", result, precision)
            st.success("Penjumlahan berhasil.")
            if show_steps:
                with st.expander("Langkah penjumlahan elemen demi elemen", expanded=True):
                    st.latex(r"c_{ij}=a_{ij}+b_{ij}")
                    if result.size <= 64:
                        for i in range(A.shape[0]):
                            for j in range(A.shape[1]):
                                st.write(f"C[{i+1},{j+1}] = {fmt_num(A[i,j])} + ({fmt_num(B[i,j])}) = **{fmt_num(result[i,j])}**")
                    else:
                        st.info("Matriks besar: gunakan rumus C[i,j] = A[i,j] + B[i,j] pada setiap posisi yang sama.")

        elif operation == "Pengurangan":
            if A.shape != B.shape:
                raise ValueError("Pengurangan memerlukan ukuran A dan B yang sama.")
            result = A - B
            show_matrix("A − B", result, precision)
            st.success("Pengurangan berhasil.")
            if show_steps:
                with st.expander("Langkah pengurangan elemen demi elemen", expanded=True):
                    st.latex(r"c_{ij}=a_{ij}-b_{ij}")
                    if result.size <= 64:
                        for i in range(A.shape[0]):
                            for j in range(A.shape[1]):
                                st.write(f"C[{i+1},{j+1}] = {fmt_num(A[i,j])} − ({fmt_num(B[i,j])}) = **{fmt_num(result[i,j])}**")
                    else:
                        st.info("Matriks besar: kurangkan setiap elemen B dari elemen A pada posisi yang sama.")

        elif operation == "Perkalian":
            if A.shape[1] != B.shape[0]:
                raise ValueError(
                    f"Kolom A ({A.shape[1]}) harus sama dengan baris B ({B.shape[0]})."
                )
            result = A @ B
            show_matrix("A × B", result, precision)
            st.success("Perkalian berhasil.")
            if show_steps and A.shape[0] <= 8 and A.shape[1] <= 8 and B.shape[1] <= 8:
                with st.expander("Langkah perkalian", expanded=True):
                    st.write("Setiap elemen hasil diperoleh dari perkalian titik baris A dengan kolom B.")
                    st.latex(r"C_{ij}=\sum_{k=1}^{n} A_{ik}B_{kj}")
                    st.write(f"Ukuran hasil: {result.shape[0]} × {result.shape[1]}.")
                    if result.size <= 36 and A.shape[1] <= 12:
                        for i in range(A.shape[0]):
                            for j in range(B.shape[1]):
                                terms = " + ".join(f"({fmt_num(A[i,k])} × {fmt_num(B[k,j])})" for k in range(A.shape[1]))
                                st.markdown(f"C[{i+1},{j+1}] = {terms} = **{fmt_num(result[i,j])}**")

        elif operation == "Transpose":
            result = A.T
            show_matrix("Transpose A (Aᵀ)", result, precision)
            st.success("Transpose berhasil.")
            if show_steps:
                with st.expander("Penjelasan", expanded=True):
                    st.write("Baris pada A menjadi kolom pada Aᵀ, dan kolom pada A menjadi baris.")

        elif operation == "Determinan":
            if A.shape[0] != A.shape[1]:
                raise ValueError("Determinan hanya dapat dihitung untuk matriks persegi.")
            det = float(np.linalg.det(A))
            st.metric("det(A)", fmt_num(det, precision))
            if show_steps:
                with st.expander("Penjelasan", expanded=True):
                    st.write("Determinan menyatakan faktor skala matriks. Ikuti rumus sesuai ukuran matriks.")
                    if A.shape == (2, 2):
                        st.latex(r"A=\begin{bmatrix}a&b\\c&d\end{bmatrix},\quad \det(A)=ad-bc")
                        st.write(f"Substitusi: det(A) = ({fmt_num(A[0,0])} × {fmt_num(A[1,1])}) − ({fmt_num(A[0,1])} × {fmt_num(A[1,0])}) = **{fmt_num(det, precision)}**")
                    elif A.shape == (3, 3):
                        st.latex(r"\det(A)=a(ei-fh)-b(di-fg)+c(dh-eg)")
                        part1 = A[0,0] * (A[1,1]*A[2,2]-A[1,2]*A[2,1])
                        part2 = A[0,1] * (A[1,0]*A[2,2]-A[1,2]*A[2,0])
                        part3 = A[0,2] * (A[1,0]*A[2,1]-A[1,1]*A[2,0])
                        st.write(f"= ({fmt_num(A[0,0])})[({fmt_num(A[1,1])}×{fmt_num(A[2,2])}) − ({fmt_num(A[1,2])}×{fmt_num(A[2,1])})] − ({fmt_num(A[0,1])})[({fmt_num(A[1,0])}×{fmt_num(A[2,2])}) − ({fmt_num(A[1,2])}×{fmt_num(A[2,0])})] + ({fmt_num(A[0,2])})[({fmt_num(A[1,0])}×{fmt_num(A[2,1])}) − ({fmt_num(A[1,1])}×{fmt_num(A[2,0])})]")
                        st.write(f"= {fmt_num(part1)} − {fmt_num(part2)} + {fmt_num(part3)} = **{fmt_num(det, precision)}**")
                    else:
                        st.write("Untuk matriks berordo lebih besar, NumPy menggunakan metode numerik berbasis eliminasi/faktorisasi. Rumus dasar untuk ordo 2×2 dan 3×3 ditampilkan di atas.")
                    st.write("Matriks A yang dihitung:")
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
                with st.expander("Metode matriks identitas: [A | I] → [I | A⁻¹]", expanded=True):
                    st.latex(r"[A \mid I] \longrightarrow [I \mid A^{-1}]")
                    st.write("1. Gabungkan A dengan matriks identitas I berukuran sama.")
                    st.write("2. Lakukan operasi baris elementer hingga bagian kiri menjadi I.")
                    st.write("3. Bagian kanan menjadi invers A⁻¹.")
                    if A.shape[0] <= 8:
                        augmented = np.hstack([A, np.eye(A.shape[0])])
                        show_matrix("[A | I] awal", augmented, precision)
                        reduced, log, _ = rref_with_snapshots(augmented)
                        show_matrix("[I | A⁻¹] akhir", reduced, precision)
                        show_row_operation_steps("Operasi baris dan matriks setelah setiap langkah", log, precision)
                    else:
                        st.info("Untuk menjaga kecepatan, matriks gabungan dan rincian operasi baris ditampilkan hanya untuk ukuran sampai 8 × 8.")

        elif operation == "Rank":
            reduced, log, rank = rref_with_snapshots(A)
            st.metric("Rank(A)", rank)
            show_matrix("Bentuk eselon baris tereduksi (RREF)", reduced, precision)
            if show_steps:
                with st.expander("Langkah operasi baris", expanded=True):
                    show_row_operation_steps("Operasi baris dan matriks setelah setiap langkah", log, precision, limit=60)

        elif operation == "Trace":
            if A.shape[0] != A.shape[1]:
                raise ValueError("Trace hanya dapat dihitung untuk matriks persegi.")
            trace = float(np.trace(A))
            st.metric("tr(A)", fmt_num(trace, precision))
            if show_steps:
                with st.expander("Penjelasan", expanded=True):
                    st.latex(r"\operatorname{tr}(A)=\sum_{i=1}^{n} a_{ii}")
                    st.write("Trace adalah jumlah elemen pada diagonal utama.")

        elif operation == "SPL - Gauss-Jordan":
            if A.shape[0] != A.shape[1]:
                raise ValueError("Untuk SPL ini, matriks koefisien A harus persegi.")
            if B.shape[0] != A.shape[0] or B.shape[1] != 1:
                raise ValueError("Matriks B untuk SPL harus berupa satu kolom dengan jumlah baris yang sama dengan A.")
            augmented = np.hstack([A, B])
            reduced, log, _ = rref_with_snapshots(augmented)
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
                with st.expander("Langkah operasi baris Gauss-Jordan", expanded=True):
                    show_row_operation_steps("Operasi baris dan matriks setelah setiap langkah", log, precision, limit=60)

        elif operation == "Doolittle Factorization":
            L, U, steps = lu_doolittle(A)
            show_matrix("Matriks L (diagonal utama = 1)", L, precision)
            show_matrix("Matriks U", U, precision)
            err = np.linalg.norm(A - L @ U)
            st.success(f"Faktorisasi selesai. Galat rekonstruksi ||A − LU|| = {err:.3e}")
            if show_steps:
                with st.expander("Rumus dan langkah Doolittle", expanded=True):
                    st.latex(r"A=LU,\quad l_{ii}=1")
                    st.write("Rumus elemen U:")
                    st.latex(r"u_{ij}=a_{ij}-\sum_{k=1}^{i-1}l_{ik}u_{kj}")
                    st.write("Rumus elemen L untuk i > j:")
                    st.latex(r"l_{ij}=\frac{a_{ij}-\sum_{k=1}^{j-1}l_{ik}u_{kj}}{u_{jj}}")
                    if steps:
                        for idx, step in enumerate(steps, start=1):
                            st.markdown(f"**{idx}.** {step}")
                    else:
                        st.info("Untuk matriks di atas 12 × 12, aplikasi menampilkan hasil dan rumus umum. Rincian substitusi angka dibatasi agar aplikasi tetap responsif.")
                    if A.shape[0] > 12:
                        st.info("Untuk matriks di atas 12 × 12, rincian langkah per elemen diringkas agar aplikasi tetap responsif.")

        elif operation == "Crout Factorization":
            L, U, steps = lu_crout(A)
            show_matrix("Matriks L", L, precision)
            show_matrix("Matriks U (diagonal utama = 1)", U, precision)
            err = np.linalg.norm(A - L @ U)
            st.success(f"Faktorisasi selesai. Galat rekonstruksi ||A − LU|| = {err:.3e}")
            if show_steps:
                with st.expander("Rumus dan langkah Crout", expanded=True):
                    st.latex(r"A=LU,\quad u_{ii}=1")
                    st.write("Rumus elemen L:")
                    st.latex(r"l_{ij}=a_{ij}-\sum_{k=1}^{j-1}l_{ik}u_{kj}")
                    st.write("Rumus elemen U untuk j > i:")
                    st.latex(r"u_{ij}=\frac{a_{ij}-\sum_{k=1}^{i-1}l_{ik}u_{kj}}{l_{ii}}")
                    if steps:
                        for idx, step in enumerate(steps, start=1):
                            st.markdown(f"**{idx}.** {step}")
                    else:
                        st.info("Untuk matriks di atas 12 × 12, aplikasi menampilkan hasil dan rumus umum. Rincian substitusi angka dibatasi agar aplikasi tetap responsif.")
                    if A.shape[0] > 12:
                        st.info("Untuk matriks di atas 12 × 12, rincian langkah per elemen diringkas agar aplikasi tetap responsif.")

        elif operation == "Cholesky Factorization":
            L, steps = cholesky_factor(A)
            show_matrix("Matriks L", L, precision)
            show_matrix("Transpose L (Lᵀ)", L.T, precision)
            err = np.linalg.norm(A - L @ L.T)
            st.success(f"Faktorisasi selesai. Galat rekonstruksi ||A − LLᵀ|| = {err:.3e}")
            if show_steps:
                with st.expander("Rumus dan langkah Cholesky", expanded=True):
                    st.markdown("### 1. Bentuk perkalian matriks terlebih dahulu")
                    st.latex(r"A=LL^T")
                    st.write("Susun L sebagai matriks segitiga bawah, lalu transposkan L untuk memperoleh Lᵀ. Kalikan L dengan Lᵀ, kemudian samakan setiap elemen hasilnya dengan elemen A.")
                    if A.shape[0] <= 5:
                        st.latex(symbolic_cholesky_product(A.shape[0]))
                    else:
                        st.info("Untuk ukuran di atas 5 × 5, perkalian simbolik penuh disembunyikan agar tampilan tetap terbaca; substitusi numerik tiap elemen tetap ditampilkan untuk ukuran sampai 12 × 12.")
                    st.markdown("### 2. Gunakan elemen A untuk mencari elemen L")
                    st.write("Urutan manualnya: cari l₁₁, kemudian elemen di bawahnya pada kolom pertama (l₂₁, l₃₁, …), lanjutkan ke l₂₂, dan seterusnya.")
                    st.latex(r"a_{ii}=l_{i1}^2+l_{i2}^2+\cdots+l_{ii}^2")
                    st.latex(r"a_{ji}=l_{j1}l_{i1}+\cdots+l_{ji}l_{ii},\quad j>i")
                    st.info("Syarat: A harus simetris dan definit positif.")
                    if steps:
                        for idx, step in enumerate(steps, start=1):
                            st.markdown(f"**{idx}.** {step}")
                    else:
                        st.info("Untuk matriks di atas 12 × 12, aplikasi menampilkan hasil dan rumus umum. Rincian substitusi angka dibatasi agar aplikasi tetap responsif.")
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
