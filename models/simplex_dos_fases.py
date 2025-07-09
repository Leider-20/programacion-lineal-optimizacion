import numpy as np

class MetodoDosFases:
    def __init__(self, coef_objetivo, restricciones, recursos, signos, modo="min"):
        self.modo = modo.lower().strip()
        self.c = np.array(coef_objetivo, dtype=float)

        # Convertimos minimización a maximización
        self.invertir_resultado = False
        if self.modo == "max":
            self.c *= -1
            self.invertir_resultado = True

        self.A = np.array(restricciones, dtype=float)
        self.b = np.array(recursos, dtype=float)
        self.signos = signos
        self.num_restricciones, self.num_variables = self.A.shape
        self._preparar_tablas()

    def _preparar_tablas(self):
        self.A_fase1 = []
        self.c_fase1 = []
        self.variables_basicas = []
        self.col_artificiales = []
        self.nuevas_variables = self.num_variables

        for i, signo in enumerate(self.signos):
            fila = list(self.A[i])
            if signo == '<=':
                fila += [1 if j == i else 0 for j in range(self.num_restricciones)]
                self.c_fase1 += [0]
                self.variables_basicas.append(self.nuevas_variables)
                self.nuevas_variables += 1
            elif signo == '>=':
                fila += [-1 if j == i else 0 for j in range(self.num_restricciones)]
                fila += [1 if j == i else 0 for j in range(self.num_restricciones)]
                self.c_fase1 += [0, 1]
                self.variables_basicas.append(self.nuevas_variables + 1)
                self.col_artificiales.append(self.nuevas_variables + 1)
                self.nuevas_variables += 2
            elif signo == '=':
                fila += [0 for _ in range(self.num_restricciones)]
                fila += [1 if j == i else 0 for j in range(self.num_restricciones)]
                self.c_fase1 += [0, 1]
                self.variables_basicas.append(self.nuevas_variables + 1)
                self.col_artificiales.append(self.nuevas_variables + 1)
                self.nuevas_variables += 2
            else:
                raise ValueError(f"Signo no válido en restricción {i}: '{signo}'")
            self.A_fase1.append(fila)

        self.A_fase1 = np.array(self.A_fase1, dtype=float)
        self.c_fase1 = np.array(self.c_fase1 + [0])

    def resolver(self):
        print("\nFase I: Encontrar solución factible")
        tabla = np.hstack([self.A_fase1, self.b.reshape(-1, 1)])
        self._resolver_simplex(tabla, self.c_fase1, fase=1)

        if any(col in self.variables_basicas for col in self.col_artificiales):
            print("\nNo hay solución factible. Variables artificiales aún en la base.")
            return

        print("\nFase II: Optimizar la función objetivo original")
        cols_validas = [i for i in range(self.nuevas_variables) if i not in self.col_artificiales]
        tabla = tabla[:, cols_validas + [-1]]
        c_fase2 = np.concatenate([self.c, np.zeros(tabla.shape[1] - len(self.c) - 1), [0]])
        self._resolver_simplex(tabla, c_fase2, fase=2)

    def _resolver_simplex(self, tabla, c, fase):
        iteracion = 0
        while True:
            print(f"\nIteración {iteracion} (Fase {fase}):")
            z_fila = c[self.variables_basicas] @ tabla[:, :-1] - c[:-1]
            z_valor = c[self.variables_basicas] @ tabla[:, -1]
            print("Zj - Cj:", np.round(z_fila, 4))
            print("Z =", z_valor)

            # Criterio de optimalidad unificado
            if np.all(z_fila <= 1e-8):
                print(f"\nÓptimo encontrado en Fase {fase}")
                if fase == 2:
                    self._mostrar_solucion(tabla)
                break

            col_pivote = np.argmax(z_fila)
            razones = [
                tabla[i, -1] / tabla[i, col_pivote] if tabla[i, col_pivote] > 0 else np.inf
                for i in range(self.num_restricciones)
            ]

            if np.all(np.isinf(razones)):
                print("\nProblema no acotado")
                return

            fila_pivote = np.argmin(razones)
            print(f"Pivote: fila {fila_pivote}, columna {col_pivote}")

            tabla[fila_pivote] /= tabla[fila_pivote, col_pivote]
            for i in range(self.num_restricciones):
                if i != fila_pivote:
                    tabla[i] -= tabla[i, col_pivote] * tabla[fila_pivote]

            self.variables_basicas[fila_pivote] = col_pivote
            iteracion += 1

    def _mostrar_solucion(self, tabla):
        x = np.zeros(tabla.shape[1] - 1)
        for i, var in enumerate(self.variables_basicas):
            if var < len(x):
                x[var] = tabla[i, -1]

        print("\nSolución óptima:")
        for i in range(self.num_variables):
            print(f"x{i+1} = {x[i]:.2f}")

        z = x[:len(self.c)] @ self.c
        if self.invertir_resultado:
            z *= -1

        print(f"Z = {z:.2f}")