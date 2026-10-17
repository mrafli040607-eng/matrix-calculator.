
import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp

st.set_page_config(
    page_title="MatrixLab",
    page_icon="🧮",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #172b40);
}
h1, h2, h3 {
    color: #67e8f9;
}
[data-testid="stSidebar"] {
    background: #111827;
}
.step-card {
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 18px;
    margin: 10px 0;
}
.step-number {
    display: inline-block;
    background: #0891b2;
    color: white;
    padding: 5px 12px;
    border-radius: 8px;
    font-weight: bold;
    margin-bottom: 10px;
}
.result-card {
    background: rgba(8, 145, 178, 0.12);
    border: 1px solid #0891b2;
    border-radius: 14px;
    padding: 18px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧮 MatrixLab")
st.caption("Kalkulator Matriks dengan Langkah Penyelesaian Terstruktur")
st.divider()


# ====================================================
# KOMPONEN TAMPILAN LANGKAH
# ====================================================

def show_matrix(title, matrix):
    st.markdown(f"**{title}**")
    arr = np.asarray(matrix)
    st.dataframe(
        pd.DataFrame(arr),
        hide_index=True,
        use_container_width=True,
        height=min(350, max(100, 35 * arr.shape[0] + 40))
    )


def show_step(number, title, explanation, formula=None, matrix=None):
    with st.container(border=True):
        st.markdown(
            f'<div class="step-number">LANGKAH {number}</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"### {title}")
        st.write(explanation)

        if formula:
            st.latex(formula)

        if matrix is not None:
            show_matrix("Matriks setelah operasi", matrix)


def show_result(title, text):
    with st.container(border=True):
        st.markdown("### ✅ " + title)
        st.markdown(text)


def input_matrix(label, rows, cols, key):
    st.markdown(f"### {label}")

    df = pd.DataFrame(
        np.zeros((rows, cols)),
        columns=[f"C{j+1}" for j in range(cols)]
    )

    edited = st.data_editor(
        df,
        key=f"{key}_{rows}_{cols}",
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        height=min(400, 35 * rows + 45)
    )
    return edited.to_numpy(dtype=float)


def check_square(A):
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Metode ini memerlukan matriks persegi.")


# ====================================================
# DETERMINAN DENGAN ELIMINASI
# ====================================================

def determinant_method(A):
    check_square(A)

    M = np.array(A, dtype=float).copy()
    n = len(M)
    det = 1.0
    sign = 1
    steps = []

    steps.append((
        "Matriks awal",
        "Tuliskan matriks A yang akan dicari determinannya.",
        None,
        M.copy()
    ))

    for i in range(n):
        pivot = i + np.argmax(np.abs(M[i:, i]))

        if abs(M[pivot, i]) < 1e-12:
            return 0.0, steps

        if pivot != i:
            M[[i, pivot]] = M[[pivot, i]]
            sign *= -1
            steps.append((
                "Pertukaran baris",
                f"Tukar baris R{i+1} dan R{pivot+1}. Tanda determinan berubah.",
                None,
                M.copy()
            ))

        pivot_value = M[i, i]
        det *= pivot_value

        steps.append((
            f"Pivot kolom {i+1}",
            f"Elemen pivot = {pivot_value:.8g}.",
            None,
            M.copy()
        ))

        for j in range(i + 1, n):
            factor = M[j, i] / pivot_value
            M[j] -= factor * M[i]

            steps.append((
                f"Eliminasi baris {j+1}",
                f"R{j+1} ← R{j+1} − ({factor:.8g})R{i+1}.",
                None,
                M.copy()
            ))

    det *= sign
    steps.append((
        "Hasil determinan",
        f"Kalikan semua pivot dan perhitungkan pertukaran baris. det(A) = {det:.10g}.",
        r"\\det(A)=(-1)^s\\prod_{i=1}^{n}u_{ii}",
        None
    ))

    return det, steps


# ====================================================
# INVERS DENGAN METODE IDENTITAS
# ====================================================

def inverse_method(A):
    check_square(A)
    M = sp.Matrix(A.tolist())
    n = M.rows

    if M.det() == 0:
        raise ValueError("Determinan nol. Invers matriks tidak ada.")

    aug = M.row_join(sp.eye(n))
    steps = []

    steps.append((
        "Bentuk matriks gabungan",
        "Gabungkan matriks A dengan matriks identitas I.",
        r"[A\\mid I]",
        np.array(aug.tolist(), dtype=object)
    ))

    rref, pivots = aug.rref()

    steps.append((
        "Operasi baris elementer",
        "Ubah blok kiri menjadi matriks identitas. Blok kanan menjadi invers A.",
        r"[A\\mid I]\\longrightarrow[I\\mid A^{-1}]",
        np.array(rref.tolist(), dtype=object)
    ))

    inverse = np.array(M.inv().tolist(), dtype=float)

    return inverse, steps


# ====================================================
# GAUSS-JORDAN
# ====================================================

def gauss_jordan_method(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matriks A harus persegi.")
    if len(b) != len(A):
        raise ValueError("Jumlah konstanta harus sama dengan jumlah baris A.")

    M = np.hstack((A, b))
    n = len(A)
    steps = []

    steps.append((
        "Matriks gabungan",
        "Gabungkan koefisien persamaan dan konstanta.",
        r"[A\\mid b]",
        M.copy()
    ))

    for i in range(n):
        pivot = i + np.argmax(np.abs(M[i:, i]))

        if abs(M[pivot, i]) < 1e-12:
            if np.linalg.matrix_rank(A) < np.linalg.matrix_rank(M):
                raise ValueError("Sistem tidak memiliki solusi.")
            raise ValueError(
                "Sistem tidak memiliki solusi tunggal. "
                "Periksa kemungkinan banyak solusi."
            )

        if pivot != i:
            M[[i, pivot]] = M[[pivot, i]]
            steps.append((
                "Pertukaran baris",
                f"R{i+1} ditukar dengan R{pivot+1}.",
                None,
                M.copy()
            ))

        value = M[i, i]
        M[i] /= value
        steps.append((
            "Normalisasi pivot",
            f"Bagikan R{i+1} dengan {value:.8g} agar pivot menjadi 1.",
            None,
            M.copy()
        ))

        for j in range(n):
            if i == j:
                continue

            factor = M[j, i]
            if abs(factor) > 1e-12:
                M[j] -= factor * M[i]
                steps.append((
                    "Eliminasi elemen",
                    f"R{j+1} ← R{j+1} − ({factor:.8g})R{i+1}.",
                    None,
                    M.copy()
                ))

    return M[:, -1], steps


# ====================================================
# DOOLITTLE
# ====================================================

def doolittle_method(A):
    check_square(A)
    A = np.array(A, dtype=float)
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))
    steps = []

    for i in range(n):
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(
                L[i, j] * U[j, k] for j in range(i)
            )
            steps.append((
                "Menghitung elemen U",
                f"U[{i+1},{k+1}] = {U[i,k]:.8g}.",
                r"u_{ik}=a_{ik}-\\sum_{j=1}^{i-1}l_{ij}u_{jk}",
                None
            ))

        if abs(U[i, i]) < 1e-12:
            raise ValueError(
                "Pivot nol. Doolittle tanpa pivoting tidak dapat dilanjutkan."
            )

        for k in range(i + 1, n):
            L[k, i] = (
                A[k, i] - sum(
                    L[k, j] * U[j, i] for j in range(i)
                )
            ) / U[i, i]

            steps.append((
                "Menghitung elemen L",
                f"L[{k+1},{i+1}] = {L[k,i]:.8g}.",
                r"l_{ki}=\\frac{a_{ki}-\\sum_{j=1}^{i-1}l_{kj}u_{ji}}{u_{ii}}",
                None
            ))

    return L, U, steps


# ====================================================
# CROUT
# ====================================================

def crout_method(A):
    check_square(A)
    A = np.array(A, dtype=float)
    n = len(A)
    L = np.zeros((n, n))
    U = np.eye(n)
    steps = []

    for j in range(n):
        for i in range(j, n):
            L[i, j] = A[i, j] - sum(
                L[i, k] * U[k, j] for k in range(j)
            )
            steps.append((
                "Menghitung elemen L",
                f"L[{i+1},{j+1}] = {L[i,j]:.8g}.",
                r"l_{ij}=a_{ij}-\\sum_{k=1}^{j-1}l_{ik}u_{kj}",
                None
            ))

        if abs(L[j, j]) < 1e-12:
            raise ValueError(
                "Pivot nol. Crout tanpa pivoting tidak dapat dilanjutkan."
            )

        for i in range(j + 1, n):
            U[j, i] = (
                A[j, i] - sum(
                    L[j, k] * U[k, i] for k in range(j)
                )
            ) / L[j, j]

            steps.append((
                "Menghitung elemen U",
                f"U[{j+1},{i+1}] = {U[j,i]:.8g}.",
                r"u_{ji}=\\frac{a_{ji}-\\sum_{k=1}^{j-1}l_{jk}u_{ki}}{l_{jj}}",
                None
            ))

    return L, U, steps


# ====================================================
# CHOLESKY
# ====================================================

def cholesky_method(A):
    check_square(A)
    A = np.array(A, dtype=float)

    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("Matriks harus simetris.")

    n = len(A)
    L = np.zeros_like(A)
    steps = []

    for i in range(n):
        for j in range(i + 1):
            value = A[i, j] - sum(
                L[i, k] * L[j, k] for k in range(j)
            )

            if i == j:
                if value <= 1e-12:
                    raise ValueError("Matriks tidak positif definit.")
                L[i, j] = np.sqrt(value)

                steps.append((
                    "Menghitung elemen diagonal L",
                    f"L[{i+1},{j+1}] = √({value:.8g}) = {L[i,j]:.8g}.",
                    r"l_{ii}=\\sqrt{a_{ii}-\\sum_{k=1}^{i-1}l_{ik}^{2}}",
                    None
                ))
            else:
                L[i, j] = value / L[j, j]

                steps.append((
                    "Menghitung elemen bawah diagonal L",
                    f"L[{i+1},{j+1}] = {value:.8g}/{L[j,j]:.8g} = {L[i,j]:.8g}.",
                    r"l_{ij}=\\frac{a_{ij}-\\sum_{k=1}^{j-1}l_{ik}l_{jk}}{l_{jj}}",
                    None
                ))

    return L, steps


# ====================================================
# SIDEBAR DAN INPUT
# ====================================================

st.sidebar.title("⚙️ Pengaturan MatrixLab")

operation = st.sidebar.selectbox(
    "Pilih metode",
    [
        "Penjumlahan",
        "Pengurangan",
        "Perkalian",
        "Transpose",
        "Determinan",
        "Invers Matriks Identitas",
        "Rank",
        "Trace",
        "SPL - Gauss-Jordan",
        "Doolittle Factorization",
        "Crout Factorization",
        "Cholesky Factorization"
    ]
)

rows_a = int(st.sidebar.number_input(
    "Baris A", 1, 100, 3
))
cols_a = int(st.sidebar.number_input(
    "Kolom A", 1, 100, 3
))

need_b = operation in ["Penjumlahan", "Pengurangan", "Perkalian"]
need_rhs = operation == "SPL - Gauss-Jordan"

rows_b, cols_b = rows_a, cols_a

if need_b:
    rows_b = int(st.sidebar.number_input(
        "Baris B", 1, 100, 3
    ))
    cols_b = int(st.sidebar.number_input(
        "Kolom B", 1, 100, 3
    ))

st.sidebar.caption(
    "Maksimum 100 × 100. Matriks besar membutuhkan waktu lebih lama."
)

left, right = st.columns(2)

with left:
    A = input_matrix("🔷 Matriks A", rows_a, cols_a, "A")

B = None

if need_b:
    with right:
        B = input_matrix("🟩 Matriks B", rows_b, cols_b, "B")

b = None

if need_rhs:
    st.subheader("Vektor konstanta b")
    b_df = st.data_editor(
        pd.DataFrame(
            np.zeros((rows_a, 1)),
            columns=["b"]
        ),
        num_rows="fixed",
        key=f"b_{rows_a}",
        use_container_width=True,
        hide_index=True
    )
    b = b_df.to_numpy(dtype=float).ravel()


# ====================================================
# HASIL DAN LANGKAH
# ====================================================

def render_steps(steps):
    st.subheader("📘 Langkah Penyelesaian")

    for i, item in enumerate(steps, start=1):
        title, explanation, formula, matrix = item

        with st.container(border=True):
            st.markdown(f"**LANGKAH {i} — {title}**")
            st.write(explanation)

            if formula:
                st.latex(formula)

            if matrix is not None:
                show_matrix("Matriks setelah langkah ini", matrix)


if st.button(
    "🚀 Hitung Matriks",
    type="primary",
    use_container_width=True
):
    try:
        st.divider()
        st.header("📊 Hasil Perhitungan")

        if operation == "Penjumlahan":
            if A.shape != B.shape:
                raise ValueError("Ukuran A dan B harus sama.")

            result = A + B
            show_matrix("A + B", result)

            steps = []
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    steps.append((
                        f"Elemen c[{i+1},{j+1}]",
                        f"{A[i,j]:g} + {B[i,j]:g} = {result[i,j]:g}.",
                        r"c_{ij}=a_{ij}+b_{ij}",
                        None
                    ))
            render_steps(steps)

        elif operation == "Pengurangan":
            if A.shape != B.shape:
                raise ValueError("Ukuran A dan B harus sama.")

            result = A - B
            show_matrix("A − B", result)

            steps = []
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    steps.append((
                        f"Elemen c[{i+1},{j+1}]",
                        f"{A[i,j]:g} − {B[i,j]:g} = {result[i,j]:g}.",
                        r"c_{ij}=a_{ij}-b_{ij}",
                        None
                    ))
            render_steps(steps)

        elif operation == "Perkalian":
            if A.shape[1] != B.shape[0]:
                raise ValueError(
                    "Jumlah kolom A harus sama dengan jumlah baris B."
                )

            result = A @ B
            show_matrix("A × B", result)

            steps = []
            for i in range(A.shape[0]):
                for j in range(B.shape[1]):
                    calculation = " + ".join(
                        f"({A[i,k]:g} × {B[k,j]:g})"
                        for k in range(A.shape[1])
                    )
                    steps.append((
                        f"Elemen c[{i+1},{j+1}]",
                        f"{calculation} = {result[i,j]:g}.",
                        r"c_{ij}=\\sum_{k=1}^{n}a_{ik}b_{kj}",
                        None
                    ))
            render_steps(steps)

        elif operation == "Transpose":
            show_matrix("Matriks A", A)
            show_matrix("Transpose A", A.T)
            render_steps([
                (
                    "Tukar baris menjadi kolom",
                    "Setiap elemen pada baris i dan kolom j dipindahkan ke baris j dan kolom i.",
                    r"(A^T)_{ij}=A_{ji}",
                    A.T
                )
            ])

        elif operation == "Determinan":
            det, steps = determinant_method(A)
            st.metric("det(A)", f"{det:.10g}")
            render_steps(steps)

        elif operation == "Invers Matriks Identitas":
            result, steps = inverse_method(A)
            show_matrix("Invers A", result)
            render_steps(steps)

        elif operation == "Rank":
            result = int(np.linalg.matrix_rank(A))
            st.metric("Rank A", result)
            render_steps([
                (
                    "Matriks awal",
                    "Amati matriks A sebelum eliminasi.",
                    None,
                    A
                ),
                (
                    "Menentukan rank",
                    f"Jumlah pivot independen matriks adalah {result}.",
                    r"\\operatorname{rank}(A)=\\text{jumlah pivot}",
                    None
                )
            ])

        elif operation == "Trace":
            check_square(A)
            diagonal = np.diag(A)
            result = np.trace(A)

            st.metric("Trace A", f"{result:g}")

            steps = []
            for i, value in enumerate(diagonal, start=1):
                steps.append((
                    f"Elemen diagonal ke-{i}",
                    f"a[{i},{i}] = {value:g}.",
                    None,
                    None
                ))

            steps.append((
                "Jumlah diagonal",
                "Jumlahkan seluruh elemen diagonal utama.",
                r"\\operatorname{tr}(A)=\\sum_{i=1}^{n}a_{ii}",
                None
            ))
            render_steps(steps)

        elif operation == "SPL - Gauss-Jordan":
            solution, steps = gauss_jordan_method(A, b)
            render_steps(steps)

            show_result(
                "Solusi SPL",
                "\n".join(
                    f"- **x{i+1} = {value:.10g}**"
                    for i, value in enumerate(solution)
                )
            )

            st.subheader("Verifikasi Ax = b")
            st.write("Ax =", A @ solution)
            st.write("b =", b)
            st.metric(
                "Galat maksimum",
                f"{np.max(np.abs(A @ solution - b)):.3e}"
            )

        elif operation == "Doolittle Factorization":
            L, U, steps = doolittle_method(A)
            show_matrix("Matriks L", L)
            show_matrix("Matriks U", U)
            st.latex(r"A=LU")
            st.metric(
                "Galat rekonstruksi",
                f"{np.linalg.norm(A-L@U):.3e}"
            )
            render_steps(steps)

        elif operation == "Crout Factorization":
            L, U, steps = crout_method(A)
            show_matrix("Matriks L", L)
            show_matrix("Matriks U", U)
            st.latex(r"A=LU")
            st.metric(
                "Galat rekonstruksi",
                f"{np.linalg.norm(A-L@U):.3e}"
            )
            render_steps(steps)

        elif operation == "Cholesky Factorization":
            L, steps = cholesky_method(A)
            show_matrix("Matriks L", L)
            show_matrix("Transpose L", L.T)
            st.latex(r"A=LL^T")
            st.metric(
                "Galat rekonstruksi",
                f"{np.linalg.norm(A-L@L.T):.3e}"
            )
            render_steps(steps)

        st.success("Perhitungan selesai.")

    except (ValueError, np.linalg.LinAlgError) as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Terjadi kesalahan: {error}")


st.divider()
st.caption("MatrixLab | Python • Streamlit • NumPy • Pandas • SymPy")
