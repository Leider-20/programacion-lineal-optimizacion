import numpy as np

class MetodoGranM:
    def __init__(self, c, A, b, signos, modo="min"):
        self.modo = modo.lower().strip()
        self.M = 1e6
        self.tipo = "min" if self.modo in ["min", "minimizar"] else "max"
        self.penalizacion = self.M if self.tipo == "min" else -self.M

        self.c_original = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.signos = signos
        self.num_restricciones, self.num_variables = self.A.shape

        self.tabla, self.c_ext, self.basicas = self._crear_tabla()

    def _crear_tabla(self):
        filas = []
        c_ext = list(self.c_original)
        basicas = []
        columna_extra = 0

        for i in range(self.num_restricciones):
            fila = list(self.A[i])
            signo = self.signos[i]

            if signo == "<=":
                fila += [1 if j == i else 0 for j in range(self.num_restricciones)]
                c_ext.append(0)
                basicas.append(self.num_variables + columna_extra)
                columna_extra += 1

            elif signo == ">=":
                fila += [-1 if j == i else 0 for j in range(self.num_restricciones)]
                fila += [1 if j == i else 0 for j in range(self.num_restricciones)]
                c_ext += [0, self.penalizacion]
                basicas.append(self.num_variables + columna_extra + 1)
                columna_extra += 2

            elif signo == "=":
                fila += [0 for _ in range(self.num_restricciones)]
                fila += [1 if j == i else 0 for j in range(self.num_restricciones)]
                c_ext += [0, self.penalizacion]
                basicas.append(self.num_variables + columna_extra + 1)
                columna_extra += 2

            else:
                raise ValueError(f"Signo no válido: {signo}")

            filas.append(fila)

        c_ext.append(0)
        tabla = np.hstack([np.array(filas, dtype=float), self.b.reshape(-1, 1)])
        return tabla, np.array(c_ext), basicas

    def resolver(self):
        iteracion = 0
        while True:
            print(f"\nIteración {iteracion}:")
            zj = self.c_ext[self.basicas] @ self.tabla[:, :-1]
            zj_cj = zj - self.c_ext[:-1]
            z_val = self.c_ext[self.basicas] @ self.tabla[:, -1]

            print("Zj - Cj:", np.round(zj_cj, 4))
            print(f"Z = {z_val:.2f}")

            if self.tipo == "min":
                optimo = np.all(zj_cj >= -1e-8)
                if optimo:
                    print("\nÓptimo encontrado")
                    return self._mostrar_solucion()
                col_piv = np.argmin(zj_cj)
            else:
                optimo = np.all(zj_cj <= 1e-8)
                if optimo:
                    print("\nÓptimo encontrado")
                    return self._mostrar_solucion()
                col_piv = np.argmax(zj_cj)

            razones = [
                self.tabla[i, -1] / self.tabla[i, col_piv] if self.tabla[i, col_piv] > 1e-8 else np.inf
                for i in range(self.num_restricciones)
            ]

            if np.all(np.isinf(razones)):
                print("\nProblema no acotado.")
                return

            fila_piv = np.argmin(razones)
            print(f"Pivote: fila {fila_piv}, columna {col_piv}")

            self.tabla[fila_piv] /= self.tabla[fila_piv, col_piv]
            for i in range(self.num_restricciones):
                if i != fila_piv:
                    self.tabla[i] -= self.tabla[i, col_piv] * self.tabla[fila_piv]

            self.basicas[fila_piv] = col_piv
            iteracion += 1

    def _mostrar_solucion(self):
        x = np.zeros(len(self.c_ext) - 1)
        for i, var in enumerate(self.basicas):
            if var < len(x):
                x[var] = self.tabla[i, -1]

        artificiales = [i for i, coef in enumerate(self.c_ext[:-1]) if abs(coef) == abs(self.penalizacion)]
        activas = [i for i in artificiales if x[i] > 1e-6]
        if activas:
            print("No hay solución factible (artificiales activas).")
            return

        z = self.c_original @ x[:self.num_variables]
        print("\nSolución óptima:")
        for i in range(self.num_variables):
            print(f"x{i+1} = {x[i]:.2f}")
        print(f"Z ({'mínima' if self.tipo == 'min' else 'máxima'}) = {z:.2f}")
