import numpy as np

class Simplex:
    def __init__(self, coef_objetivo, restricciones, recursos, modo="min"):
        self.modo = modo.strip().lower()
        self.c = np.array(coef_objetivo, dtype=float)

        # Si el usuario seleccionó "min", invertimos los coeficientes
        self.invertir_resultado = False
        if self.modo == "max":
            self.c *= -1
            self.invertir_resultado = True

        self.A = np.array(restricciones, dtype=float)
        self.b = np.array(recursos, dtype=float)

        self.num_restricciones, self.num_variables = self.A.shape
        self._preparar_tabla()

    def _preparar_tabla(self):
        identidad = np.eye(self.num_restricciones)
        self.tabla = np.hstack([self.A, identidad, self.b.reshape(-1, 1)])
        self.c_ext = np.concatenate([self.c, np.zeros(self.num_restricciones), [0]])
        self.basicas = list(range(self.num_variables, self.num_variables + self.num_restricciones))

    def resolver(self):
        iteracion = 0
        while True:
            print(f"\nIteración {iteracion}:")
            self._imprimir_tabla()
            self._imprimir_variables_basicas()

            num_columnas = self.tabla.shape[1]
            num_variables_total = num_columnas - 1

            zj = self.c_ext[self.basicas] @ self.tabla[:, :num_variables_total]
            zj_cj = zj - self.c_ext[:num_variables_total]
            z_valor = self.c_ext[self.basicas] @ self.tabla[:, -1]

            print("\nZj - Cj:", np.round(zj_cj, 4))
            print(f"Valor actual de Z = {z_valor:.2f}")

            if np.all(zj_cj <= 1e-8):
                print("\nÓptimo encontrado")
                self._mostrar_solucion()
                break

            col_entrada = np.argmax(zj_cj)

            razones = [
                self.tabla[i, -1] / self.tabla[i, col_entrada] if self.tabla[i, col_entrada] > 0 else np.inf
                for i in range(self.num_restricciones)
            ]

            if np.all(np.isinf(razones)):
                print("\nProblema no acotado")
                return

            fila_salida = np.argmin(razones)
            print(f"Pivote: fila {fila_salida}, columna {col_entrada} (x{col_entrada+1})")

            self._pivoteo(fila_salida, col_entrada)
            self.basicas[fila_salida] = col_entrada
            iteracion += 1

    def _pivoteo(self, fila, columna):
        self.tabla[fila] /= self.tabla[fila, columna]
        for i in range(self.num_restricciones):
            if i != fila:
                self.tabla[i] -= self.tabla[i, columna] * self.tabla[fila]

    def _mostrar_solucion(self):
        x = np.zeros(self.num_variables + self.num_restricciones)
        for i, var in enumerate(self.basicas):
            if var < len(x):
                x[var] = self.tabla[i, -1]

        print("\nSolución óptima:")
        for i in range(self.num_variables):
            print(f"x{i+1} = {x[i]:.2f}")

        z = self.c_ext[self.basicas] @ self.tabla[:, -1]
        if self.invertir_resultado:
            z *= -1
        print(f"Z = {z:.2f}")

    def _imprimir_tabla(self):
        print("Tabla simplex actual:")
        print(np.round(self.tabla, 4))

    def _imprimir_variables_basicas(self):
        print("\nVariables básicas actuales:")
        for i, var in enumerate(self.basicas):
            nombre = f"x{var+1}" if var < self.num_variables else f"s{var - self.num_variables + 1}"
            valor = self.tabla[i, -1]
            print(f"{nombre} = {valor:.2f}")