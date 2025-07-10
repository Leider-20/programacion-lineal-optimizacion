import numpy as np

class SimplexAlgebraico:
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

        self.A_ext = np.hstack([self.A, np.eye(self.num_restricciones)])
        self.variables_basicas = list(range(self.num_variables, self.num_variables + self.num_restricciones))
        self.variables_no_basicas = list(range(self.num_variables))
        self.B = self.A_ext[:, self.variables_basicas]
        self.N = self.A_ext[:, self.variables_no_basicas]

    def resolver(self):
        iteracion = 0
        while True:
            print(f"\nIteración {iteracion}:")
            B_inv = np.linalg.inv(self.B)
            x_b = B_inv @ self.b

            c_b = np.array([self.c[i] if i < len(self.c) else 0 for i in self.variables_basicas])
            z = c_b @ x_b

            print("Variables básicas:", self.variables_basicas)
            print("x_b:", x_b)
            print(f"Valor actual de Z: {z:.2f}")

            c_n = np.array([self.c[i] if i < len(self.c) else 0 for i in self.variables_no_basicas])
            z_n = c_b @ B_inv @ self.N
            evaluadores = np.array(z_n - c_n)
            print("Zj - Cj:", np.round(evaluadores, 4))

            # Criterio de optimalidad (siempre maximizando)
            if np.all(evaluadores <= 1e-8):
                print("\nÓptimo encontrado")
                self.mostrar_resultado(x_b, z)
                break

            # Variable entrante (mayor evaluador positivo)
            col_entrante = np.argmax(evaluadores)

            direccion = B_inv @ self.N[:, col_entrante]
            if np.all(direccion <= 0):
                print("\nProblema no acotado")
                break

            razones = [x_b[i] / direccion[i] if direccion[i] > 0 else np.inf for i in range(len(direccion))]
            fila_saliente = np.argmin(razones)

            # Intercambiar variables
            self.variables_basicas[fila_saliente], self.variables_no_basicas[col_entrante] = \
                self.variables_no_basicas[col_entrante], self.variables_basicas[fila_saliente]

            self.B = self.A_ext[:, self.variables_basicas]
            self.N = self.A_ext[:, self.variables_no_basicas]
            iteracion += 1

    def mostrar_resultado(self, x_b, z):
        solucion = np.zeros(self.num_variables + self.num_restricciones)
        for i, var in enumerate(self.variables_basicas):
            solucion[var] = x_b[i]

        print("\nSolución óptima:")
        for i in range(self.num_variables):
            print(f"x{i+1} = {solucion[i]:.2f}")

        if self.invertir_resultado:
            z *= -1

        print(f"Z = {z:.2f}")