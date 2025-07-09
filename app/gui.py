import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sys
import io

from models.simplex import Simplex
from models.simplex_tabular import SimplexTabular
from models.simplex_algebraico import SimplexAlgebraico
from models.metodo_grafico import MetodoGrafico
from models.simplex_m_grande import MetodoMGrande
from models.simplex_dos_fases import MetodoDosFases
from models.simplex_revisado import SimplexRevisado
from analisis_sensibilidad import AnalisisSensibilidad


class AplicacionPL:
    def __init__(self, raiz):
        self.raiz = raiz
        self.raiz.title("PL Solver - UdeA")
        self.raiz.geometry("700x650")
        self._crear_widgets()

    def _crear_widgets(self):
        # Método de resolución
        ttk.Label(self.raiz, text="Método de resolución:").pack(pady=5)
        self.metodo_var = tk.StringVar()
        self.combo_metodo = ttk.Combobox(self.raiz, textvariable=self.metodo_var, state="readonly")
        self.combo_metodo['values'] = (
            "Método gráfico",
            "Simplex",
            "Simplex tabular",
            "Simplex algebraico",
            "Simplex revisado",
            "M grande",
            "Dos fases"
        )
        self.combo_metodo.current(0)
        self.combo_metodo.pack()

        # NUEVO: Modo de optimización
        ttk.Label(self.raiz, text="Modo de optimización:").pack(pady=5)
        self.modo_var = tk.StringVar()
        self.combo_modo = ttk.Combobox(self.raiz, textvariable=self.modo_var, state="readonly")
        self.combo_modo['values'] = ("Maximizar", "Minimizar")
        self.combo_modo.current(0)
        self.combo_modo.pack()

        # Función objetivo
        ttk.Label(self.raiz, text="Función objetivo (separar por comas):").pack(pady=5)
        self.entrada_objetivo = ttk.Entry(self.raiz, width=100)
        self.entrada_objetivo.pack()

        # Restricciones
        ttk.Label(self.raiz, text="Restricciones (una por línea, coeficientes separados por comas, sin RHS):").pack(pady=5)
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
            modo_opt = "max" if self.combo_modo.current() == 0 else "min"
            c = list(map(float, self.entrada_objetivo.get().split(',')))
            b = list(map(float, self.entrada_b.get().split(',')))
            A = [list(map(float, fila.strip().split(','))) for fila in self.caja_restricciones.get("1.0", tk.END).strip().split('\n')]

            if len(A) != len(b):
                messagebox.showerror("Error", "El número de restricciones y recursos no coincide")
                return

            self.salida.delete("1.0", tk.END)

            if metodo == "Simplex":
                solucionador = Simplex(c, A, b, modo=modo_opt)
                self._redirigir_salida(solucionador.resolver)

            elif metodo == "Simplex tabular":
                solucionador = SimplexTabular(c, A, b, modo=modo_opt)
                self._redirigir_salida(solucionador.resolver)

            elif metodo == "Simplex algebraico":
                solucionador = SimplexAlgebraico(c, A, b, modo=modo_opt)
                self._redirigir_salida(solucionador.resolver)

            elif metodo == "Simplex revisado":
                solucionador = SimplexRevisado(c, A, b, modo=modo_opt)
                self._redirigir_salida(solucionador.resolver)

            elif metodo == "Método gráfico":
                if len(c) != 2:
                    messagebox.showerror("Error", "El método gráfico solo funciona con 2 variables")
                    return
                solucionador = MetodoGrafico(c, A, b, modo=modo_opt)
                self._redirigir_salida(solucionador.resolver)

            elif metodo == "M grande":
                signos = self.entrada_signos.get().strip().split(',')
                solucionador = MetodoMGrande(c, A, b, signos, modo=modo_opt)
                self._redirigir_salida(solucionador.resolver)

            elif metodo == "Dos fases":
                signos = self.entrada_signos.get().strip().split(',')
                solucionador = MetodoDosFases(c, A, b, signos, modo=modo_opt)
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
