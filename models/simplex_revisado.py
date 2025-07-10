import numpy as np

class SimplexRevisado:
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

        # Ampliamos la matriz A con variables de holgura
        self.A_ext = np.hstack([self.A, np.eye(self.num_restricciones)])
        self.c_ext = np.concatenate([self.c, np.zeros(self.num_restricciones)])

        # Inicializamos las variables básicas y no básicas
        self.basicas = list(range(self.num_variables, self.num_variables + self.num_restricciones))
        self.no_basicas = list(range(self.num_variables))

    def resolver(self):
        iteracion = 0
        while True:
            print(f"\nIteración {iteracion}:")
            B = self.A_ext[:, self.basicas]
            N = self.A_ext[:, self.no_basicas]
            c_b = self.c_ext[self.basicas]
            c_n = self.c_ext[self.no_basicas]

            B_inv = np.linalg.inv(B)
            x_b = B_inv @ self.b
            z = c_b @ x_b

            print("Variables básicas:", self.basicas)
            print("x_b:", x_b)
            print(f"Valor de Z: {z:.2f}")

            evaluadores = c_b @ B_inv @ N - c_n
            print("Zj - Cj:", np.round(evaluadores, 4))

            # Criterio de optimalidad (siempre maximizando)
            if np.all(evaluadores <= 1e-8):
                print("\nSolución óptima encontrada")
                self.mostrar_solucion(x_b, z)
                break

            # Variable entrante (mayor valor positivo)
            j = np.argmax(evaluadores)
            direccion = B_inv @ N[:, j]

            if np.all(direccion <= 0):
                print("\nProblema no acotado")
                break

            razones = [x_b[i] / direccion[i] if direccion[i] > 0 else np.inf for i in range(len(direccion))]
            i = np.argmin(razones)

            print(f"Variable que entra: x{self.no_basicas[j]+1}")
            print(f"Variable que sale: x{self.basicas[i]+1}")
            print(f"Razón mínima: {razones[i]:.2f}")

            # Intercambio de variables
            self.basicas[i], self.no_basicas[j] = self.no_basicas[j], self.basicas[i]
            iteracion += 1

    def mostrar_solucion(self, x_b, z):
        solucion = np.zeros(len(self.c_ext))
        for i, var in enumerate(self.basicas):
            solucion[var] = x_b[i]

        print("\nSolución óptima:")
        for i in range(self.num_variables):
            print(f"x{i+1} = {solucion[i]:.2f}")

        if self.invertir_resultado:
            z *= -1

        print(f"Z = {z:.2f}")