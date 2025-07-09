import numpy as np

class AnalisisSensibilidad:
    def __init__(self, A, b, c, basicas):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)
        self.basicas = basicas

    def analizar(self):
        B = self.A[:, self.basicas]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            print("La matriz base no es invertible. No se puede hacer análisis de sensibilidad.")
            return

        x_b = B_inv @ self.b
        c_b = self.c[self.basicas]
        z = c_b @ x_b

        print("\n--- Análisis de Sensibilidad ---")
        print("Solución básica actual:")
        for i, var in enumerate(self.basicas):
            print(f"x{var+1} = {x_b[i]:.2f}")
        print(f"Valor óptimo de Z: {z:.2f}\n")

        # Precios sombra (valores duales)
        print("Precios sombra (valores duales):")
        precios_sombra = c_b @ B_inv
        for i, valor in enumerate(precios_sombra):
            print(f"Restricción {i+1}: {valor:.2f}")

        # Rango de variación de b (lado derecho)
        print("\nRangos de variación de b (recursos):")
        for i in range(len(self.b)):
            e = np.zeros_like(self.b)
            e[i] = 1
            cambio = B_inv @ e

            limite_inf = -np.inf
            limite_sup = np.inf
            for j in range(len(cambio)):
                if cambio[j] != 0:
                    ratio = x_b[j] / cambio[j]
                    if cambio[j] > 0:
                        limite_sup = min(limite_sup, ratio)
                    else:
                        limite_inf = max(limite_inf, ratio)

            lim_inf_texto = f"{self.b[i] + limite_inf:.2f}" if limite_inf != -np.inf else "-inf"
            lim_sup_texto = f"{self.b[i] + limite_sup:.2f}" if limite_sup != np.inf else "inf"

            print(f"b{i+1}: entre {lim_inf_texto} y {lim_sup_texto}")
