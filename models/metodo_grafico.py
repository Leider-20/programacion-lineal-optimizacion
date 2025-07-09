import numpy as np
import matplotlib.pyplot as plt

class MetodoGrafico:
    def __init__(self, coef_objetivo, restricciones, recursos, modo="max"):
        self.modo = modo.lower()
        self.c = np.array(coef_objetivo, dtype=float)
        self.A = np.array(restricciones, dtype=float)
        self.b = np.array(recursos, dtype=float)

    def resolver(self):
        if self.A.shape[1] != 2:
            print("\nEl método gráfico solo funciona con dos variables.")
            return

        x1 = np.linspace(0, 100, 400)
        restricciones_graficas = []

        for i in range(len(self.A)):
            if self.A[i, 1] != 0:
                x2 = (self.b[i] - self.A[i, 0] * x1) / self.A[i, 1]
            else:
                x2 = np.full_like(x1, np.inf)
            restricciones_graficas.append(x2)

        plt.figure(figsize=(8, 6))

        for i, x2 in enumerate(restricciones_graficas):
            plt.plot(x1, x2, label=f"Restricción {i+1}")

        plt.xlim(0, max(x1))
        plt.ylim(0, 100)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Método gráfico - Región factible")
        plt.grid(True)

        # Encontrar vértices factibles
        vertices = self.obtener_vertices()
        vertices = [np.array(v) for v in vertices if v is not None and len(v) == 2]
        vertices_viables = [v for v in vertices if np.all(self.A @ v <= self.b + 1e-6) and np.all(v >= 0)]

        if not vertices_viables:
            print("\nNo hay región factible.")
            return

        valores_z = [self.c @ v for v in vertices_viables]

        if self.modo == "max":
            z_opt = max(valores_z)
            idx = valores_z.index(z_opt)
        else:
            z_opt = min(valores_z)
            idx = valores_z.index(z_opt)

        optimo = vertices_viables[idx]

        for v in vertices_viables:
            plt.plot(v[0], v[1], 'ko')
        plt.plot(optimo[0], optimo[1], 'ro', label="Óptimo")
        plt.legend()
        plt.show()

        print("\nVértices viables:")
        for i, v in enumerate(vertices_viables):
            print(f"Punto {i+1}: x1 = {v[0]:.2f}, x2 = {v[1]:.2f}, Z = {valores_z[i]:.2f}")

        tipo = "máxima" if self.modo == "max" else "mínima"
        print(f"\nSolución {tipo}: x1 = {optimo[0]:.2f}, x2 = {optimo[1]:.2f}, Z = {z_opt:.2f}")

    def obtener_vertices(self):
        vertices = []
        n = len(self.A)
        for i in range(n):
            for j in range(i + 1, n):
                A_sub = np.array([self.A[i], self.A[j]])
                b_sub = np.array([self.b[i], self.b[j]])
                try:
                    punto = np.linalg.solve(A_sub, b_sub)
                    vertices.append(punto)
                except np.linalg.LinAlgError:
                    continue

        for i in range(n):
            if self.A[i, 0] != 0:
                x1 = self.b[i] / self.A[i, 0]
                if x1 >= 0:
                    vertices.append([x1, 0])
            if self.A[i, 1] != 0:
                x2 = self.b[i] / self.A[i, 1]
                if x2 >= 0:
                    vertices.append([0, x2])

        vertices.append([0, 0])
        return vertices
