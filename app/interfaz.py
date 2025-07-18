import tkinter as tk
from tkinter import ttk, messagebox

from models.simplex_tabular import SimplexTabular
from models.metodo_grafico import MetodoGrafico
from models.simplex_gran_m import MetodoGranM
from models.simplex_revisado import SimplexRevisado
from models.analisis_sensibilidad import AnalisisSensibilidad


class AplicacionPL:
    def __init__(self, raiz):
        self.raiz = raiz
        self.raiz.title("Programa para resolver problemas de PL")
        self.raiz.geometry("700x650")
        self._crear_elementos()

    def _crear_elementos(self):
        # Método de resolución
        ttk.Label(self.raiz, text="Método de resolución:").pack(pady=5)
        self.metodo_var = tk.StringVar()
        self.combo_metodo = ttk.Combobox(self.raiz, textvariable=self.metodo_var, state="readonly")
        self.combo_metodo['values'] = (
            "Método gráfico",
            "Simplex tabular",
            "Simplex revisado",
            "M grande"
        )
        self.combo_metodo.current(0)
        self.combo_metodo.pack()

        # Función objetivo
        ttk.Label(self.raiz, text="Coeficientes de la función objetivo (separar por comas):").pack(pady=5)
        self.entrada_objetivo = ttk.Entry(self.raiz, width=100)
        self.entrada_objetivo.pack()

        # Restricciones
        ttk.Label(self.raiz, text="Restricciones (una por línea, solo coeficientes, separados por comas):").pack(pady=5)
        self.caja_restricciones = tk.Text(self.raiz, height=6, width=100)
        self.caja_restricciones.pack()

        # Recursos
        ttk.Label(self.raiz, text="Recursos (lado derecho de cada restricción, separados por comas):").pack(pady=5)
        self.entrada_b = ttk.Entry(self.raiz, width=100)
        self.entrada_b.pack()

        # Signos
        ttk.Label(self.raiz, text="Signos de las restricciones (<=, >=, =):").pack(pady=5)
        self.entrada_signos = ttk.Entry(self.raiz, width=100)
        self.entrada_signos.pack()

        # Para análisis de sensibilidad
        ttk.Label(self.raiz, text="Variables básicas (solo para análisis de sensibilidad):").pack(pady=5)
        self.entrada_basicas = ttk.Entry(self.raiz, width=100)
        self.entrada_basicas.pack()

        ttk.Button(self.raiz, text="Resolver", command=self.resolver).pack(pady=15)
        self.salida = tk.Text(self.raiz, height=20, width=100)
        self.salida.pack()

    def resolver(self):
        try:
            metodo = self.metodo_var.get()
            c = list(map(float, self.entrada_objetivo.get().split(',')))
            b = list(map(float, self.entrada_b.get().split(',')))
            A = [list(map(float, fila.strip().split(','))) for fila in self.caja_restricciones.get("1.0", tk.END).strip().split('\n')]

            if len(A) != len(b):
                messagebox.showerror("Error", "El número de restricciones y recursos no coincide")
                return

            self.salida.delete("1.0", tk.END)

            if metodo == "Simplex tabular":
                solucionador = SimplexTabular(c, A, b)
                self._redirigir_salida(solucionador.resolver)

            elif metodo == "Simplex revisado":
                solucionador = SimplexRevisado(c, A, b)
                self._redirigir_salida(solucionador.resolver)

            elif metodo == "Método gráfico":
                if len(c) != 2:
                    messagebox.showerror("Error", "El método gráfico solo funciona con 2 variables")
                    return
                solucionador = MetodoGrafico(c, A, b)
                self._redirigir_salida(solucionador.resolver)

            elif metodo == "M grande":
                signos = self.entrada_signos.get().strip().split(',')
                solucionador = MetodoGranM(c, A, b, signos)
                self._redirigir_salida(solucionador.resolver)

            elif metodo == "Análisis de sensibilidad":
                basicas = list(map(int, self.entrada_basicas.get().strip().split(',')))
                solucionador = AnalisisSensibilidad(A, b, c, basicas)
                self._redirigir_salida(solucionador.analizar)

        except Exception as e:
            messagebox.showerror("Error", f"Datos inválidos: {e}")

    def _redirigir_salida(self, funcion):
        import sys, io
        buffer = io.StringIO()
        sys.stdout = buffer
        funcion()
        sys.stdout = sys.__stdout__
        self.salida.insert(tk.END, buffer.getvalue())
