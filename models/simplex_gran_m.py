import numpy as np

class MetodoGranM:
    def __init__(self, coef_objetivo, restricciones, recursos, signos, modo="max"):
        self.modo = modo.lower().strip()
        if self.modo in ("max", "maximizar"):
            self.sentido = "max"
            self.M = -1e6
        elif self.modo in ("min", "minimizar"):
            self.sentido = "min"
            self.M =  1e6
        else:
            raise ValueError("modo debe ser 'max'/'maximizar' o 'min'/'minimizar'")

        self.c_orig = np.array(coef_objetivo, float)
        self.A = np.array(restricciones, float)
        self.b = np.array(recursos, float)
        self.signos = signos

        m, n = self.A.shape
        # contar variables extras
        n_h = sum(1 for s in signos if s == "<=")
        n_e = sum(1 for s in signos if s == ">=")
        n_a = sum(1 for s in signos if s in (">=", "="))
        total_extra = n_h + n_e + n_a

        # preparar c_ext: [coef_x ...] + [0]*n_h + [0]*n_e + [M]*n_a + [0 para LD]
        self.c_ext = np.concatenate([
            self.c_orig,
            np.zeros(n_h + n_e),
            np.full(n_a, self.M),
            [0]
        ])

        # construir tabla inicial
        A_ext = []
        basicas = []
        idx_h = idx_e = idx_a = 0

        for i, s in enumerate(signos):
            fila = np.zeros(n + total_extra)
            fila[:n] = self.A[i]

            # columna base de holgura
            if s == "<=":
                col = n + idx_h
                fila[col] = 1
                basicas.append(col)
                idx_h += 1

            # exceso + artificial
            elif s == ">=":
                col_e = n + n_h + idx_e
                col_a = n + n_h + n_e + idx_a
                fila[col_e] = -1
                fila[col_a] =  1
                basicas.append(col_a)
                idx_e += 1
                idx_a += 1

            # solo artificial
            elif s == "=":
                col_a = n + n_h + n_e + idx_a
                fila[col_a] = 1
                basicas.append(col_a)
                idx_a += 1

            else:
                raise ValueError(f"signo inválido: {s}")

            A_ext.append(fila)

        # tabla simplex con LD al final
        self.tabla = np.hstack([np.array(A_ext), self.b.reshape(-1,1)])
        self.basicas = basicas
        self.num_restricciones, _ = self.tabla.shape

    def resolver(self):
        it = 0
        while True:
            print(f"\n--- Iteración {it} ---")
            zj   = self.c_ext[self.basicas] @ self.tabla[:,:-1]
            zj_c = zj - self.c_ext[:-1]
            z_val= self.c_ext[self.basicas] @ self.tabla[:,-1]

            print("Zj - Cj:", np.round(zj_c,4))
            print("Z =", z_val)

            # criterio de optimalidad
            if self.sentido == "max":
                if np.all(zj_c <= 1e-8):
                    break
                col_piv = np.argmax(zj_c)
            else:
                if np.all(zj_c >= -1e-8):
                    break
                col_piv = np.argmin(zj_c)

            # prueba de acotamiento
            razones = [
                self.tabla[i,-1] / self.tabla[i,col_piv]
                if self.tabla[i,col_piv] > 0 else np.inf
                for i in range(self.num_restricciones)
            ]
            if np.all(np.isinf(razones)):
                print("Problema no acotado")
                return

            fila_piv = np.argmin(razones)
            print(f"Pivote → fila {fila_piv}, columna {col_piv}")

            # pivotear
            self.tabla[fila_piv] /= self.tabla[fila_piv,col_piv]
            for i in range(self.num_restricciones):
                if i != fila_piv:
                    self.tabla[i] -= self.tabla[i,col_piv] * self.tabla[fila_piv]

            self.basicas[fila_piv] = col_piv
            it += 1

        # mostrar solución
        x = np.zeros(len(self.c_ext)-1)
        for i, var in enumerate(self.basicas):
            if var < len(x):
                x[var] = self.tabla[i,-1]

        # verificar artificiales
        art_cols = [i for i, v in enumerate(self.c_ext[:-1]) if abs(v)==abs(self.M)]
        if any(x[c]>1e-6 for c in art_cols):
            print("Sin solución factible (artificiales activas).")
            return

        z_opt = self.c_orig @ x[:self.A.shape[1]]
        print("\n→ Solución óptima:")
        for j in range(self.A.shape[1]):
            print(f" x{j+1} = {x[j]:.2f}")
        print(f" Z = {z_opt:.2f}")