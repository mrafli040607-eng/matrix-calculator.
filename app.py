
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
    color: #f1f5f9;
}
h1, h2, h3 {
    color: #67e8f9;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.title("🧮 MatrixLab")
st.caption("Kalkulator Matriks dan Penyelesaian Langkah demi Langkah")
st.divider()


# =====================================================
# FUNGSI TAMPILAN
# =====================================================

def show_matrix(title, matrix, key=None):
    """Menampilkan matriks sebagai tabel."""
    st.markdown(f"**{title}**")
    arr = np.asarray(matrix)
    st.dataframe(
        pd.DataFrame(arr),
        hide_index=True,
        use_container_width=True,
        height=min(420, 38 * arr.shape[0] + 40)
    )


def show_steps(title, steps):
    """Menampilkan setiap langkah secara terpisah."""
    with st.expander(f"📘 {title}", expanded=True):
        for i, step in enumerate(steps, start=1):
            st.markdown(f"**Langkah {i}.** {step['text']}")
            if "matrix" in step:
                show_matrix("Matriks setelah operasi", step["matrix"])
            if "formula" in step:
                st.latex(step["formula"])


def input_matrix(label, rows, cols, key):
    """Input matriks berbentuk kotak-kotak."""
    st.subheader(label)

    df = pd.DataFrame(
        np.zeros((rows, cols), dtype=float),
        columns=[f"C{j + 1}" for j in range(cols)]
    )

    edited = st.data_editor(
        df,
        key=f"{key}_{rows}_{cols}",
        num_rows="fixed",
        use_container_width=True,
        height=min(420, 38 * rows + 40),
        hide_index=True
    )

    return edited.to_numpy(dtype=float)


def validate_square(A):
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Metode ini memerlukan matriks persegi.")


# =====================================================
# OPERASI DASAR DENGAN LANGKAH
# =====================================================

def determinant_steps(A):
    validate_square(A)
    M = sp.Matrix(A.tolist())
    steps = [
        {
            "text": "Tuliskan matriks A yang akan dicari determinannya.",
            "matrix": np.array(M.tolist(), dtype=object)
        }
    ]

    det = M.det()
    steps.append({
        "text": (
            "Hitung determinan menggunakan eliminasi simbolik. "
            "Untuk matriks 2×2, rumusnya ad - bc; "
            "untuk ukuran lebih besar, perhitungan diperluas."
        ),
        "formula": r"\det(A)"
    })
    steps.append({
        "text": f"Hasil determinan adalah {det}."
    })

    return det, steps


def inverse_steps(A):
    validate_square(A)
    M = sp.Matrix(A.tolist())
    n = M.rows

    if M.det() == 0:
        raise ValueError("Matriks singular; invers tidak ada.")

    aug = M.row_join(sp.eye(n))
    steps = [{
        "text": "Gabungkan A dengan matriks identitas I.",
        "matrix": np.array(aug.tolist(), dtype=object),
        "formula": r"[A\\mid I]"
    }]

    rref_matrix, pivots = aug.rref()
    steps.append({
        "text": (
            "Lakukan operasi baris elementer hingga bagian kiri "
            "menjadi matriks identitas."
        ),
        "matrix": np.array(rref_matrix.tolist(), dtype=object),
        "formula": r"[A\\mid I]\\longrightarrow[I\\mid A^{-1}]"
    })

    inv = M.inv()
    steps.append({
        "text": "Bagian kanan matriks gabungan adalah invers A."
    })

    return np.array(inv.tolist(), dtype=float), steps


def rank_steps(A):
    M = sp.Matrix(A.tolist())
    rref_matrix, pivots = M.rref()
    rank = len(pivots)

    steps = [
        {
            "text": "Tuliskan matriks A.",
            "matrix": np.array(M.tolist(), dtype=object)
        },
        {
            "text": (
                "Gunakan eliminasi Gauss-Jordan untuk mendapatkan "
                "bentuk eselon baris tereduksi."
            ),
            "matrix": np.array(rref_matrix.tolist(), dtype=object)
        },
        {
            "text": (
                f"Terdapat {rank} pivot. Oleh karena itu, "
                f"rank(A) = {rank}."
            )
        }
    ]

    return rank, steps


def trace_steps(A):
    validate_square(A)
    diagonal = np.diag(A)
    total = float(np.sum(diagonal))

    steps = [{
        "text": "Ambil elemen diagonal utama matriks A.",
        "formula": r"\\operatorname{tr}(A)=\\sum_{i=1}^{n}a_{ii}"
    }]

    for i, value in enumerate(diagonal, start=1):
        steps.append({
            "text": f"Elemen diagonal ke-{i} adalah {value:g}."
        })

    steps.append({
        "text": f"Jumlah elemen diagonal = {total:g}."
    })

    return total, steps


# =====================================================
# GAUSS-JORDAN DENGAN REKAMAN SETIAP OPERASI
# =====================================================

def gauss_jordan_steps(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matriks A harus persegi.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Jumlah elemen b harus sama dengan baris A.")

    n = A.shape[0]
    M = np.hstack((A, b))
    steps = [{
        "text": "Bentuk matriks gabungan [A | b].",
        "matrix": M.copy()
    }]

    for i in range(n):
        pivot = i + np.argmax(np.abs(M[i:, i]))

        if abs(M[pivot, i]) < 1e-12:
            if np.linalg.matrix_rank(A) < np.linalg.matrix_rank(M):
                raise ValueError("SPL tidak memiliki solusi.")
            raise ValueError(
                "SPL tidak memiliki solusi tunggal. "
                "Mungkin terdapat banyak solusi."
            )

        if pivot != i:
            M[[i, pivot]] = M[[pivot, i]]
            steps.append({
                "text": f"Tukar baris R{i+1} dengan R{pivot+1}.",
                "matrix": M.copy()
            })

        pivot_value = M[i, i]
        M[i] = M[i] / pivot_value
        steps.append({
            "text": (
                f"Jadikan pivot baris R{i+1} bernilai 1 "
                f"dengan membagi baris oleh {pivot_value:g}."
            ),
            "matrix": M.copy()
        })

        for j in range(n):
            if j == i:
                continue

            factor = M[j, i]
            if abs(factor) > 1e-12:
                M[j] = M[j] - factor * M[i]
                steps.append({
                    "text": (
                        f"Eliminasi elemen baris R{j+1}, kolom {i+1}: "
                        f"R{j+1} ← R{j+1} − ({factor:g})R{i+1}."
                    ),
                    "matrix": M.copy()
                })

    solution = M[:, -1]
    steps.append({
        "text": "Matriks kiri sudah menjadi identitas. Kolom terakhir adalah solusi.",
        "matrix": M.copy()
    })

    return solution, steps


# =====================================================
# FAKTORISASI LU DENGAN LANGKAH
# =====================================================

def doolittle_steps(A):
    validate_square(A)
    A = np.array(A, dtype=float)
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))
    steps = [{
        "text": "Mulai dengan L memiliki diagonal 1 dan U bernilai nol.",
        "matrix": A.copy()
    }]

    for i in range(n):
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(
                L[i, j] * U[j, k] for j in range(i)
            )
            steps.append({
                "text": f"Hitung U[{i+1},{k+1}] = {U[i,k]:g}."
            })

        if abs(U[i, i]) < 1e-12:
            raise ValueError(
                "Pivot nol ditemukan. Doolittle tanpa pivoting gagal."
            )

        for k in range(i + 1, n):
            L[k, i] = (
                A[k, i] - sum(
                    L[k, j] * U[j, i] for j in range(i)
                )
            ) / U[i, i]
            steps.append({
                "text": f"Hitung L[{k+1},{i+1}] = {L[k,i]:g}."
            })

    steps.append({
        "text": "Faktorisasi selesai: A = LU.",
        "matrix": L @ U
    })

    return L, U, steps


def crout_steps(A):
    validate_square(A)
    A = np.array(A, dtype=float)
    n = len(A)
    L = np.zeros((n, n))
    U = np.eye(n)
    steps = [{
        "text": "Mulai dengan U memiliki diagonal 1.",
        "matrix": A.copy()
    }]

    for j in range(n):
        for i in range(j, n):
            L[i, j] = A[i, j] - sum(
                L[i, k] * U[k, j] for k in range(j)
            )
            steps.append({
                "text": f"Hitung L[{i+1},{j+1}] = {L[i,j]:g}."
            })

        if abs(L[j, j]) < 1e-12:
            raise ValueError(
                "Pivot nol ditemukan. Crout tanpa pivoting gagal."
            )

        for i in range(j + 1, n):
            U[j, i] = (
                A[j, i] - sum(
                    L[j, k] * U[k, i] for k in range(j)
                )
            ) / L[j, j]
            steps.append({
                "text": f"Hitung U[{j+1},{i+1}] = {U[j,i]:g}."
            })

    steps.append({
        "text": "Faktorisasi selesai: A = LU.",
        "matrix": L @ U
    })

    return L, U, steps


def cholesky_steps(A):
    validate_square(A)
    A = np.array(A, dtype=float)

    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("Matriks harus simetris.")

    n = len(A)
    L = np.zeros_like(A)
    steps = [{
        "text": "Periksa bahwa A simetris dan siapkan matriks L.",
        "matrix": A.copy()
    }]

    for i in range(n):
        for j in range(i + 1):
            value = A[i, j] - sum(
                L[i, k] * L[j, k] for k in range(j)
            )

            if i == j:
                if value <= 1e-12:
                    raise ValueError(
                        "Matriks tidak positif definit."
                    )
                L[i, j] = np.sqrt(value)
                steps.append({
                    "text": (
                        f"L[{i+1},{j+1}] = √({value:g}) "
                        f"= {L[i,j]:g}."
                    )
                })
            else:
                L[i, j] = value / L[j, j]
                steps.append({
                    "text": f"L[{i+1},{j+1}] = {L[i,j]:g}."
                })

    steps.append({
        "text": "Faktorisasi selesai: A = LLᵀ.",
        "matrix": L @ L.T
    })

    return L, steps


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Pengaturan")

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

rows_a = st.sidebar.number_input(
    "Baris A", min_value=1, max_value=100, value=3
)
cols_a = st.sidebar.number_input(
    "Kolom A", min_value=1, max_value=100, value=3
)

need_b = operation in ["Penjumlahan", "Pengurangan", "Perkalian"]
need_rhs = operation == "SPL - Gauss-Jordan"

rows_b, cols_b = int(rows_a), int(cols_a)

if need_b:
    rows_b = st.sidebar.number_input(
        "Baris B", min_value=1, max_value=100, value=3
    )
    cols_b = st.sidebar.number_input(
        "Kolom B", min_value=1, max_value=100, value=3
    )

st.sidebar.info(
    "Ukuran maksimum 100 × 100. "
    "Perhitungan matriks besar membutuhkan waktu lebih lama."
)


# =====================================================
# INPUT MATRIKS
# =====================================================

left, right = st.columns(2)

with left:
    A = input_matrix(
        "🔷 Matriks A",
        int(rows_a),
        int(cols_a),
        "A"
    )

B = None

if need_b:
    with right:
        B = input_matrix(
            "🟩 Matriks B",
            int(rows_b),
            int(cols_b),
            "B"
        )

b = None

if need_rhs:
    st.subheader("Vektor konstanta b")

    b_df = st.data_editor(
        pd.DataFrame(
            np.zeros((int(rows_a), 1)),
            columns=["b"]
        ),
        num_rows="fixed",
        use_container_width=True,
        key=f"vector_b_{int(rows_a)}",
        hide_index=True,
        height=min(420, 38 * int(rows_a) + 40)
    )

    b = b_df.to_numpy(dtype=float).ravel()


# =====================================================
# EKSEKUSI OPERASI
# =====================================================

if st.button(
    "🚀 Hitung dan Tampilkan Langkah",
    type="primary",
    use_container_width=True
):
    try:
        st.divider()
        st.header("📊 Hasil dan Langkah Penyelesaian")

        if operation == "Penjumlahan":
            if A.shape != B.shape:
                raise ValueError("Ukuran A dan B harus sama.")

            result = A + B
            show_matrix("Matriks A", A)
            show_matrix("Matriks B", B)
            show_matrix("Hasil A + B", result)

            st.markdown("**Rumus elemen:**")
            st.latex(r"c_{ij}=a_{ij}+b_{ij}")

            steps = []
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    steps.append({
                        "text": (
                            f"c[{i+1},{j+1}] = "
                            f"{A[i,j]:g} + {B[i,j]:g} "
                            f"= {result[i,j]:g}"
                        )
                    })
            show_steps("Langkah penjumlahan", steps)

        elif operation == "Pengurangan":
            if A.shape != B.shape:
                raise ValueError("Ukuran A dan B harus sama.")

            result = A - B
            show_matrix("Hasil A − B", result)

            steps = []
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    steps.append({
                        "text": (
                            f"c[{i+1},{j+1}] = "
                            f"{A[i,j]:g} − {B[i,j]:g} "
                            f"= {result[i,j]:g}"
                        )
                    })
            show_steps("Langkah pengurangan", steps)

        elif operation == "Perkalian":
            if A.shape[1] != B.shape[0]:
                raise ValueError(
                    "Kolom A harus sama dengan baris B."
                )

            result = A @ B
            show_matrix("Hasil A × B", result)

            steps = []
            for i in range(A.shape[0]):
                for j in range(B.shape[1]):
                    products = [
                        f"({A[i,k]:g}×{B[k,j]:g})"
                        for k in range(A.shape[1])
                    ]
                    steps.append({
                        "text": (
                            f"c[{i+1},{j+1}] = "
                            + " + ".join(products)
                            + f" = {result[i,j]:g}"
                        )
                    })
            show_steps("Langkah perkalian", steps)

        elif operation == "Transpose":
            result = A.T
            show_matrix("Matriks A", A)
            show_matrix("Transpose A", result)
            show_steps("Langkah transpose", [
                {
                    "text": (
                        "Tukar baris menjadi kolom dan kolom menjadi baris."
                    ),
                    "formula": r"(A^T)_{ij}=A_{ji}"
                }
            ])

        elif operation == "Determinan":
            det, steps = determinant_steps(A)
            show_matrix("Matriks A", A)
            st.metric("det(A)", str(det))
            show_steps("Langkah determinan", steps)

        elif operation == "Invers Matriks Identitas":
            result, steps = inverse_steps(A)
            show_matrix("Invers A", result)
            show_steps("Langkah metode matriks identitas", steps)

        elif operation == "Rank":
            result, steps = rank_steps(A)
            st.metric("Rank matriks A", result)
            show_steps("Langkah menentukan rank", steps)

        elif operation == "Trace":
            result, steps = trace_steps(A)
            st.metric("Trace A", f"{result:g}")
            show_steps("Langkah menentukan trace", steps)

        elif operation == "SPL - Gauss-Jordan":
            solution, steps = gauss_jordan_steps(A, b)
            show_steps("Langkah Gauss-Jordan", steps)

            st.subheader("Solusi variabel")
            st.dataframe(pd.DataFrame({
                "Variabel": [
                    f"x{i+1}" for i in range(len(solution))
                ],
                "Nilai": solution
            }), use_container_width=True)

            residual = A @ solution - b
            st.subheader("Verifikasi Ax = b")
            st.write("Nilai Ax:", A @ solution)
            st.write("Nilai b:", b)
            st.metric(
                "Galat maksimum",
                f"{np.max(np.abs(residual)):.3e}"
            )

        elif operation == "Doolittle Factorization":
            L, U, steps = doolittle_steps(A)
            show_matrix("Matriks L", L)
            show_matrix("Matriks U", U)
            show_matrix("Verifikasi LU", L @ U)
            st.latex(r"A=LU")
            st.metric(
                "Galat rekonstruksi",
                f"{np.linalg.norm(A - L @ U):.3e}"
            )
            show_steps("Langkah Doolittle", steps)

        elif operation == "Crout Factorization":
            L, U, steps = crout_steps(A)
            show_matrix("Matriks L", L)
            show_matrix("Matriks U", U)
            show_matrix("Verifikasi LU", L @ U)
            st.latex(r"A=LU")
            st.metric(
                "Galat rekonstruksi",
                f"{np.linalg.norm(A - L @ U):.3e}"
            )
            show_steps("Langkah Crout", steps)

        elif operation == "Cholesky Factorization":
            L, steps = cholesky_steps(A)
            show_matrix("Matriks L", L)
            show_matrix("Transpose L (Lᵀ)", L.T)
            show_matrix("Verifikasi LLᵀ", L @ L.T)
            st.latex(r"A=LL^T")
            st.metric(
                "Galat rekonstruksi",
                f"{np.linalg.norm(A - L @ L.T):.3e}"
            )
            show_steps("Langkah Cholesky", steps)

        st.success("Perhitungan selesai.")

    except (ValueError, np.linalg.LinAlgError) as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Terjadi kesalahan: {error}")


st.divider()
st.caption("MatrixLab • Python • Streamlit • NumPy • Pandas • SymPy")
