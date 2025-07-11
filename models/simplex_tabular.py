import numpy as np

class SimplexTabular:
    def __init__(self, coef_objetivo, restricciones, recursos):
        self.c_original = np.array(coef_objetivo, dtype=float)
        self.A = np.array(restricciones, dtype=float)
        self.b = np.array(recursos, dtype=float)
        self.num_restricciones, self.num_variables = self.A.shape

        # Detectar si es minimización
        self.invertir_resultado = False
        if np.all(self.c_original >= 0):
            self.c = -self.c_original
            self.invertir_resultado = True
        else:
            self.c = self.c_original

        # Construir tabla inicial con variables de holgura
        self.tabla = np.hstack([self.A, np.eye(self.num_restricciones), self.b.reshape(-1, 1)])
        self.c_extendida = np.concatenate([self.c, np.zeros(self.num_restricciones + 1)])  # +1 para LD
        self.base = list(range(self.num_variables, self.num_variables + self.num_restricciones))

    def imprimir_tabla(self):
        print("\nTabla Simplex actual:")
        print(np.round(self.tabla, 4))

    def resolver(self):
        iteracion = 0
        while True:
            print(f"\nIteración {iteracion}:")
            self.imprimir_tabla()

            fila_z = self.c_extendida[self.base] @ self.tabla[:, :-1] - self.c_extendida[:-1]
            fila_z = np.array(fila_z)
            valor_z = self.c_extendida[self.base] @ self.tabla[:, -1]

            print("Zj - Cj:", np.round(fila_z, 4))
            print("Valor actual de Z:", valor_z)

            # Criterio de optimalidad unificado (siempre maximizando)
            if np.all(fila_z <= 1e-8):
                print("\nÓptimo encontrado")
                break

            # Selección de variable entrante (mayor valor positivo)
            columna_pivote = np.argmax(fila_z)
            razones = []

            for i in range(self.num_restricciones):
                if self.tabla[i, columna_pivote] > 0:
                    razon = self.tabla[i, -1] / self.tabla[i, columna_pivote]
                else:
                    razon = np.inf
                razones.append(razon)

            if np.all(np.isinf(razones)):
                print("\nProblema no acotado")
                return

            fila_pivote = np.argmin(razones)
            print(f"Pivote: fila {fila_pivote}, columna {columna_pivote}")

            self.pivotear(fila_pivote, columna_pivote)
            self.base[fila_pivote] = columna_pivote
            iteracion += 1

        self.imprimir_solucion()

    def pivotear(self, fila, columna):
        self.tabla[fila] /= self.tabla[fila, columna]
        for i in range(self.num_restricciones):
            if i != fila:
                self.tabla[i] -= self.tabla[i, columna] * self.tabla[fila]

    def imprimir_solucion(self):
        x = np.zeros(self.num_variables + self.num_restricciones)
        for i, var in enumerate(self.base):
            x[var] = self.tabla[i, -1]

        print("\nSolución óptima:")
        for i in range(self.num_variables):
            print(f"x{i+1} = {x[i]:.2f}")

        z = self.c_extendida[self.base] @ self.tabla[:, -1]
        if self.invertir_resultado:
            z *= -1

        print(f"Z = {z:.2f}")