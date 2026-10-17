
    
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

