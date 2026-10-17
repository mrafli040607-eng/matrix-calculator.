
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
    background: linear-gradient(135deg, #101827, #172b40);
    color: #f1f5f9;
}
h1, h2, h3 {
    color: #67e8f9;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
div[data-testid="stDataFrame"],
div[data-testid="stDataEditor"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧮 MatrixLab")
st.caption("Kalkulator Matriks • Python + Streamlit")
st.divider()


# =====================================================
# FUNGSI MATRIKS
# =====================================================

def matrix_input(label, rows, cols, key):
    st.subheader(label)

    # Membuat input matriks berbentuk tabel
    data = pd.DataFrame(
        np.zeros((rows, cols), dtype=float),
        columns=[f"C{j + 1}" for j in range(cols)]
    )

    edited = st.data_editor(
        data,
        num_rows="fixed",
        use_container_width=True,
        key=f"{key}_{rows}_{cols}",
        height=min(400, 38 * rows + 40)
    )

    return edited.to_numpy(dtype=float)


def show_matrix(title, matrix):
    st.subheader(title)
    st.dataframe(
        pd.DataFrame(np.asarray(matrix)),
        use_container_width=True
    )


def gauss_jordan(A, b):
    """Menyelesaikan Ax=b menggunakan eliminasi Gauss-Jordan."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matriks A harus persegi.")

    if A.shape[0] != b.shape[0]:
        raise ValueError("Jumlah baris A dan b harus sama.")

    augmented = np.hstack((A, b))
    n = A.shape[0]

    for i in range(n):
        pivot = i + np.argmax(np.abs(augmented[i:, i]))

        if abs(augmented[pivot, i]) < 1e-12:
            raise ValueError(
                "Sistem tidak memiliki solusi tunggal."
            )

        augmented[[i, pivot]] = augmented[[pivot, i]]
        augmented[i] = augmented[i] / augmented[i, i]

        for j in range(n):
            if j != i:
                augmented[j] -= augmented[j, i] * augmented[i]

    return augmented[:, -1], augmented


def doolittle(A):
    """Faktorisasi A = LU dengan diagonal L bernilai 1."""
    A = np.array(A, dtype=float)
    n = A.shape[0]

    if A.shape != (n, n):
        raise ValueError("Matriks harus persegi.")

    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(
                L[i, j] * U[j, k] for j in range(i)
            )

        if abs(U[i, i]) < 1e-12 and i < n - 1:
            raise ValueError(
                "Pivot nol ditemukan. Doolittle tanpa pivoting gagal."
            )

        for k in range(i + 1, n):
            if abs(U[i, i]) < 1e-12:
                raise ValueError("Pivot nol ditemukan.")
            L[k, i] = (
                A[k, i] - sum(
                    L[k, j] * U[j, i] for j in range(i)
                )
            ) / U[i, i]

    return L, U


def crout(A):
    """Faktorisasi A = LU dengan diagonal U bernilai 1."""
    A = np.array(A, dtype=float)
    n = A.shape[0]

    L = np.zeros((n, n))
    U = np.eye(n)

    for j in range(n):
        for i in range(j, n):
            L[i, j] = A[i, j] - sum(
                L[i, k] * U[k, j] for k in range(j)
            )

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

    return L, U


def cholesky(A):
    """Faktorisasi A = LLᵀ."""
    A = np.array(A, dtype=float)

    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("Matriks harus simetris.")

    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Matriks harus positif definit agar Cholesky berhasil."
        ) from exc

    return L


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Pengaturan")

operation = st.sidebar.selectbox(
    "Pilih Operasi",
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

st.sidebar.subheader("Ukuran Matriks A")

rows_a = st.sidebar.number_input(
    "Baris A", min_value=1, max_value=100,
    value=3, step=1
)
cols_a = st.sidebar.number_input(
    "Kolom A", min_value=1, max_value=100,
    value=3, step=1
)

needs_b = operation in [
    "Penjumlahan", "Pengurangan", "Perkalian"
]

needs_rhs = operation == "SPL - Gauss-Jordan"

rows_b, cols_b = rows_a, cols_a

if needs_b:
    st.sidebar.subheader("Ukuran Matriks B")
    rows_b = st.sidebar.number_input(
        "Baris B", min_value=1, max_value=100,
        value=3, step=1
    )
    cols_b = st.sidebar.number_input(
        "Kolom B", min_value=1, max_value=100,
        value=3, step=1
    )

st.sidebar.caption(
    "Ukuran maksimum 100 × 100. "
    "Matriks besar mungkin memerlukan waktu lebih lama."
)


# =====================================================
# INPUT
# =====================================================

col1, col2 = st.columns(2)

with col1:
    A = matrix_input(
        "🔷 Matriks A",
        int(rows_a),
        int(cols_a),
        "matrix_A"
    )

B = None

if needs_b:
    with col2:
        B = matrix_input(
            "🟩 Matriks B",
            int(rows_b),
            int(cols_b),
            "matrix_B"
        )

b = None

if needs_rhs:
    st.subheader("Vektor konstanta b")
    b_df = st.data_editor(
        pd.DataFrame(
            np.zeros((int(rows_a), 1)),
            columns=["b"]
        ),
        num_rows="fixed",
        use_container_width=True,
        key=f"vector_b_{int(rows_a)}",
        height=min(400, 38 * int(rows_a) + 40)
    )
    b = b_df.to_numpy(dtype=float).ravel()


# =====================================================
# HITUNG
# =====================================================

if st.button("🚀 Hitung Matriks", type="primary",
             use_container_width=True):

    try:
        st.divider()
        st.header("📊 Hasil Perhitungan")

        if operation == "Penjumlahan":
            if A.shape != B.shape:
                st.error("Ukuran A dan B harus sama.")
            else:
                show_matrix("A + B", A + B)

        elif operation == "Pengurangan":
            if A.shape != B.shape:
                st.error("Ukuran A dan B harus sama.")
            else:
                show_matrix("A − B", A - B)

        elif operation == "Perkalian":
            if A.shape[1] != B.shape[0]:
                st.error(
                    "Kolom A harus sama dengan baris B."
                )
            else:
                show_matrix("A × B", A @ B)

        elif operation == "Transpose":
            show_matrix("Transpose A (Aᵀ)", A.T)

        elif operation == "Determinan":
            if A.shape[0] != A.shape[1]:
                st.error("Determinan hanya untuk matriks persegi.")
            else:
                st.metric(
                    "det(A)",
                    f"{np.linalg.det(A):.8g}"
                )

        elif operation == "Invers Matriks Identitas":
            if A.shape[0] != A.shape[1]:
                st.error("Invers hanya untuk matriks persegi.")
            elif abs(np.linalg.det(A)) < 1e-12:
                st.error("Matriks singular, sehingga invers tidak ada.")
            else:
                n = A.shape[0]
                augmented = np.hstack([A.copy(), np.eye(n)])
                show_matrix("[A | I]", augmented)

                inverse = np.linalg.inv(A)
                show_matrix("A⁻¹", inverse)

                st.info(
                    "Metode matriks identitas: ubah [A | I] "
                    "menjadi [I | A⁻¹] menggunakan operasi baris elementer."
                )

        elif operation == "Rank":
            st.metric("Rank A", int(np.linalg.matrix_rank(A)))

        elif operation == "Trace":
            if A.shape[0] != A.shape[1]:
                st.error("Trace hanya untuk matriks persegi.")
            else:
                st.metric("Trace A", f"{np.trace(A):.8g}")

        elif operation == "SPL - Gauss-Jordan":
            solution, augmented = gauss_jordan(A, b)
            show_matrix("[A | b] setelah Gauss-Jordan", augmented)
            st.subheader("Nilai variabel")
            st.dataframe(
                pd.DataFrame({
                    "Variabel": [
                        f"x{i + 1}" for i in range(len(solution))
                    ],
                    "Nilai": solution
                }),
                use_container_width=True
            )

        elif operation == "Doolittle Factorization":
            if A.shape[0] != A.shape[1]:
                st.error("Doolittle memerlukan matriks persegi.")
            else:
                L, U = doolittle(A)
                show_matrix("Matriks L", L)
                show_matrix("Matriks U", U)
                st.latex(r"A = LU")
                st.metric(
                    "Galat rekonstruksi",
                    f"{np.linalg.norm(A - L @ U):.3e}"
                )

        elif operation == "Crout Factorization":
            if A.shape[0] != A.shape[1]:
                st.error("Crout memerlukan matriks persegi.")
            else:
                L, U = crout(A)
                show_matrix("Matriks L", L)
                show_matrix("Matriks U", U)
                st.latex(r"A = LU")
                st.metric(
                    "Galat rekonstruksi",
                    f"{np.linalg.norm(A - L @ U):.3e}"
                )

        elif operation == "Cholesky Factorization":
            if A.shape[0] != A.shape[1]:
                st.error("Cholesky memerlukan matriks persegi.")
            else:
                L = cholesky(A)
                show_matrix("Matriks L", L)
                show_matrix("Transpose L (Lᵀ)", L.T)
                st.latex(r"A = LL^T")
                st.metric(
                    "Galat rekonstruksi",
                    f"{np.linalg.norm(A - L @ L.T):.3e}"
                )

        st.success("Perhitungan selesai.")

    except (ValueError, np.linalg.LinAlgError) as error:
        st.error(f"Perhitungan tidak dapat dilakukan: {error}")
    except Exception as error:
        st.error(f"Terjadi kesalahan: {error}")


st.divider()
st.caption(
    "MatrixLab • Dibangun dengan Python, Streamlit, NumPy, dan Pandas"
)
