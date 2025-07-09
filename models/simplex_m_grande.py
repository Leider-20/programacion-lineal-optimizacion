import numpy as np

class MetodoMGrande:
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
        self.M = 1e6  # Penalización muy grande
        self.num_restricciones, self.num_variables = self.A.shape

        self.A_ext, self.c_ext, self.variables_basicas = self._crear_forma_ampliada()

    def _crear_forma_ampliada(self):
        A_ext = []
        c_ext = list(self.c)
        variables_basicas = []
        var_adicional = self.num_variables

        for i, signo in enumerate(self.signos):
            fila = list(self.A[i])
            if signo == '<=':
                fila += [1 if j == i else 0 for j in range(self.num_restricciones)]
                c_ext += [0]
                variables_basicas.append(var_adicional)
                var_adicional += 1
            elif signo == '>=':
                fila += [-1 if j == i else 0 for j in range(self.num_restricciones)]
                fila += [1 if j == i else 0 for j in range(self.num_restricciones)]
                c_ext += [0, self.M]
                variables_basicas.append(var_adicional + 1)
                var_adicional += 2
            elif signo == '=':
                fila += [0 for _ in range(self.num_restricciones)]
                fila += [1 if j == i else 0 for j in range(self.num_restricciones)]
                c_ext += [0, self.M]
                variables_basicas.append(var_adicional + 1)
                var_adicional += 2
            else:
                raise ValueError(f"Signo no válido en la restricción {i}: '{signo}'")

            A_ext.append(fila)

        return np.array(A_ext, dtype=float), np.array(c_ext + [0]), variables_basicas

    def resolver(self):
        tabla = np.hstack([self.A_ext, self.b.reshape(-1, 1)])
        iteracion = 0

        while True:
            print(f"\nIteración {iteracion}:")
            z_fila = self.c_ext[self.variables_basicas] @ tabla[:, :-1] - self.c_ext[:-1]
            z_valor = self.c_ext[self.variables_basicas] @ tabla[:, -1]
            print("Zj - Cj:", np.round(z_fila, 4))
            print("Z =", z_valor)

            # Criterio de optimalidad unificado
            if np.all(z_fila <= 1e-8):
                print("\nSolución óptima encontrada")
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
        x = np.zeros(len(self.c_ext) - 1)
        for i, var in enumerate(self.variables_basicas):
            if var < len(x):
                x[var] = tabla[i, -1]

        print("\nSolución óptima:")
        for i in range(self.num_variables):
            print(f"x{i+1} = {x[i]:.2f}")

        z = self.c_ext[self.variables_basicas] @ tabla[:, -1]
        if self.invertir_resultado:
            z *= -1

        print(f"Z = {z:.2f}")