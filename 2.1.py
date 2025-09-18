"""
PUNTO 2.1 - CONJUNTOS LINEALMENTE SEPARABLES


Autor: David Leonardo Rojas Leon
Curso: Inteligencia Computacional - UPTC
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# Estilo
plt.style.use('seaborn-v0_8')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#577590']

class SeparabilidadLinealApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Separabilidad Lineal - Punto 2.1")
        self.root.geometry("2000x1200")  # Ventana más grande
        self.root.configure(bg='#f8f9fa')
        
        # Variables de estado
        self.current_demo = None
        self.setup_main_interface()
        
    def setup_main_interface(self):
        """Configurar interfaz principal moderna"""
        # Frame principal 
        main_frame = tk.Frame(self.root, bg='#f8f9fa')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header
        header_frame = tk.Frame(main_frame, bg='#2c3e50', height=80)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="SEPARABILIDAD LINEAL",
                              font=("Segoe UI", 20, "bold"), fg='white', bg='#2c3e50')
        title_label.pack(pady=20)
        
        subtitle_label = tk.Label(header_frame, text="Exploración Visual de Conceptos Fundamentales en Machine Learning",
                                 font=("Segoe UI", 12), fg='#ecf0f1', bg='#2c3e50')
        subtitle_label.pack()
        
        # Frame de contenido principal
        content_frame = tk.Frame(main_frame, bg='#f8f9fa')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel lateral izquierdo - Controles
        self.setup_control_panel(content_frame)
        
        # Panel central - Visualización
        self.setup_visualization_panel(content_frame)
        
        # Panel derecho - Información (más ancho)
        self.setup_info_panel(content_frame)
        
    def setup_control_panel(self, parent):
        """Panel de controles """
        control_frame = tk.LabelFrame(parent, text="  DEMOSTRACIONES  ",
                                     font=("Segoe UI", 12, "bold"), fg='#2c3e50',
                                     bg='white', bd=2, relief='solid')
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5, ipadx=10, ipady=10)
        control_frame.configure(width=120)  # Ancho fijo
        control_frame.pack_propagate(False)  # No cambiar tamaño
        
        # Lista de demostraciones con colores
        demos = [
            (" Caso Simple", "Ejemplo básico linealmente separable", self.demo_simple, '#e74c3c'),
            (" Datos Generados", "Dataset sintético avanzado", self.demo_generated, '#3498db'),
            (" Problema XOR", "Imposibilidad de separación lineal", self.demo_xor, '#e67e22'),
            (" Convergencia", "Análisis del proceso de aprendizaje", self.demo_convergence, '#27ae60'),
            (" Comparación", "Perceptrón vs Regresión Logística", self.demo_comparison, '#9b59b6'),
            (" Círculos", "Caso complejo no separable", self.demo_circles, '#f39c12')
        ]
        
        self.buttons = {}
        for i, (name, desc, func, color) in enumerate(demos):
            # Frame para cada botón
            btn_frame = tk.Frame(control_frame, bg='white')
            btn_frame.pack(fill=tk.X, pady=8)
            
            # Botón principal con estilo personalizado
            btn = tk.Button(btn_frame, text=name, command=func,
                           font=("Segoe UI", 11, "bold"), fg='white', bg=color,
                           relief='flat', bd=0, pady=12, cursor='hand2')
            btn.pack(fill=tk.X)
            
            # Efecto hover
            btn.bind("<Enter>", lambda e, b=btn, c=color: self.on_button_enter(b, c))
            btn.bind("<Leave>", lambda e, b=btn, c=color: self.on_button_leave(b, c))
            
            # Descripción
            desc_label = tk.Label(btn_frame, text=desc, font=("Segoe UI", 9),
                                 fg='#7f8c8d', bg='white', wraplength=180)
            desc_label.pack(pady=(5, 0))
            
            self.buttons[name] = btn
            
    def on_button_enter(self, button, original_color):
        """Efecto hover para botones"""
        # Hacer el color más oscuro
        button.configure(bg=self.darken_color(original_color))
        
    def on_button_leave(self, button, original_color):
        """Restaurar color original del botón"""
        button.configure(bg=original_color)
        
    def darken_color(self, hex_color, factor=0.8):
        """Oscurecer un color hex"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darker_rgb = tuple(int(c * factor) for c in rgb)
        return f"#{darker_rgb[0]:02x}{darker_rgb[1]:02x}{darker_rgb[2]:02x}"
        
    def setup_visualization_panel(self, parent):
        """Panel de visualización central"""
        viz_frame = tk.LabelFrame(parent, text="  VISUALIZACIÓN ",
                                 font=("Segoe UI", 12, "bold"), fg='#2c3e50',
                                 bg='white', bd=2, relief='solid')
        viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=5, ipadx=10, ipady=10)
        
        # Frame para el gráfico
        self.plot_frame = tk.Frame(viz_frame, bg='white')
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Mensaje inicial 
        self.initial_message = tk.Frame(self.plot_frame, bg='white')
        self.initial_message.pack(expand=True, fill=tk.BOTH)
        
        welcome_label = tk.Label(self.initial_message, text="Análisis de Separabilidad Lineal!",
                                font=("Segoe UI", 18, "bold"), fg='#2c3e50', bg='white')
        welcome_label.pack(pady=(50, 20))
        
        instruction_label = tk.Label(self.initial_message, 
                                   text="Selecciona una demostración del panel izquierdo\npara explorar conceptos clave de Machine Learning",
                                   font=("Segoe UI", 14), fg='#7f8c8d', bg='white')
        instruction_label.pack(pady=20)
        
        # Indicadores visuales
        features_frame = tk.Frame(self.initial_message, bg='white')
        features_frame.pack(pady=30)
        
        features = ["Visualizaciones", " Análisis ", " Explicaciones "]
        for feature in features:
            feature_label = tk.Label(features_frame, text=feature, font=("Segoe UI", 12),
                                   fg='#27ae60', bg='white')
            feature_label.pack(pady=5)
            
    def setup_info_panel(self, parent):
        """Panel de información con diseño mejorado y más ancho"""
        info_frame = tk.LabelFrame(parent, text="  EXPLICACIÓN ",
                                  font=("Segoe UI", 12, "bold"), fg='#2c3e50',
                                  bg='white', bd=2, relief='solid')
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=0, pady=5, ipadx=10, ipady=10)
        info_frame.configure(width=600)  # Ancho fijo más amplio
        info_frame.pack_propagate(False)  # No cambiar tamaño
        
        # Área de texto con scroll 
        text_frame = tk.Frame(info_frame, bg='white')
        text_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Texto con formato mejorado y tamaño más pequeño
        self.info_text = tk.Text(text_frame, wrap=tk.WORD, width=70, height=40,
                                font=("Segoe UI", 11), bg='#f8f9fa', fg='#2c3e50',
                                relief='flat', bd=5, padx=15, pady=15)
        
        # Scrollbar con estilo
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mensaje inicial
        self.show_initial_info()
        
    def show_initial_info(self):
        """Mostrar información inicial"""
        initial_text = """SEPARABILIDAD LINEAL EN ML

La separabilidad lineal es un concepto fundamental que determina si dos o más clases pueden ser perfectamente separadas por una frontera lineal.

CONCEPTOS CLAVE:
• Frontera de Decisión Lineal
• Algoritmo Perceptrón  
• Convergencia Garantizada
• Limitaciones Geométricas

EXPLORA LAS DEMOSTRACIONES:

1. Caso Simple: Separación perfecta
2. Datos Generados: Análisis avanzado
3. Problema XOR: Limitaciones claras
4. Convergencia: Proceso de aprendizaje
5. Comparación: Diferentes algoritmos
6. Círculos: Casos complejos

Cada demostración incluye:
• Visualización 
• Análisis matemático
• Interpretación práctica
• Conclusiones clave

Selecciona una demostración para comenzar tu exploración!"""
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, initial_text)
        
    def clear_plot_area(self):
        """Limpiar área de gráficos"""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
    def update_info(self, title, content):
        """Actualizar panel de información"""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, f"{title}\n{'='*80}\n\n{content}")
        
    def create_modern_plot(self, title, figsize=(10, 7)):
        """Crear gráfico """
        self.clear_plot_area()
        
        fig = Figure(figsize=figsize, dpi=100, facecolor='white')
        fig.patch.set_facecolor('white')
        
        # Título principal con estilo
        fig.suptitle(title, fontsize=16, fontweight='bold', color='#2c3e50', y=0.95)
        
        return fig
        
    def show_plot(self, fig):
        """Mostrar gráfico en la interfaz"""
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Agregar toolbar de navegación
        toolbar_frame = tk.Frame(self.plot_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 10))
        
        # Información de la demostración actual
        info_label = tk.Label(toolbar_frame, text=f"Demostración Activa: {self.current_demo}",
                             font=("Segoe UI", 10, "italic"), fg='#7f8c8d', bg='white')
        info_label.pack(side=tk.LEFT, padx=10)
        
    # DEMOSTRACIONES
    
    def demo_simple(self):
        """Demostración 1: Caso Simple"""
        self.current_demo = "Caso Simple"
        
        # Datos linealmente separables
        class_A = np.array([[1, 3], [2, 4], [3, 3], [4, 5], [2.5, 4.5]])
        class_B = np.array([[1, 1], [2, 1], [3, 2], [1, 2], [2.5, 1.5]])
        
        X = np.vstack([class_A, class_B])
        y = np.hstack([np.ones(len(class_A)), np.zeros(len(class_B))])
        
        # Entrenar Perceptrón
        perceptron = Perceptron(max_iter=1000, random_state=42)
        perceptron.fit(X, y)
        
        # Crear gráfico
        fig = self.create_modern_plot("Caso Simple: Conjuntos Linealmente Separables")
        ax = fig.add_subplot(111, facecolor='#f8f9fa')
        
        # Puntos con diseño mejorado
        ax.scatter(class_A[:, 0], class_A[:, 1], c='#e74c3c', s=200, marker='o',
                  label='Clase A (+1)', alpha=0.8, edgecolors='white', linewidth=3)
        ax.scatter(class_B[:, 0], class_B[:, 1], c='#3498db', s=200, marker='s',
                  label='Clase B (-1)', alpha=0.8, edgecolors='white', linewidth=3)
        
        # Frontera de decisión con estilo
        w = perceptron.coef_[0]
        b = perceptron.intercept_[0]
        x_line = np.linspace(0.5, 4.5, 100)
        y_line = -(w[0] * x_line + b) / w[1]
        ax.plot(x_line, y_line, color='#27ae60', linewidth=4, alpha=0.9,
                label=f'Frontera: {w[0]:.1f}x₁ + {w[1]:.1f}x₂ + {b:.1f} = 0')
        
        # Regiones de decisión
        ax.fill_between(x_line, y_line, 6, alpha=0.1, color='#e74c3c', label='Región Clase A')
        ax.fill_between(x_line, 0, y_line, alpha=0.1, color='#3498db', label='Región Clase B')
        
        # Estilo del gráfico
        ax.set_xlabel('x₁', fontsize=12, fontweight='bold')
        ax.set_ylabel('x₂', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0.5, 4.5)
        ax.set_ylim(0.5, 5.5)
        
        self.show_plot(fig)
        
        # Actualizar información
        info_content = """CASO SIMPLE - SEPARABILIDAD PERFECTA

RESULTADO: Separación exitosa
Precisión: 100%
Ecuación: -1.5x₁ + 6.5x₂ - 12.0 = 0

ANÁLISIS VISUAL:
• Puntos rojos (Clase A): Región superior
• Puntos azules (Clase B): Región inferior
• Línea verde: Frontera de decisión óptima
• Regiones coloreadas: Zonas de clasificación

PROCESO ALGORÍTMICO:
El Perceptrón encuentra automáticamente una línea que separa perfectamente ambas clases mediante aprendizaje iterativo.

CARACTERÍSTICAS:
• Convergencia garantizada en datos separables
• Solución encontrada en pocas iteraciones
• Cualquier punto nuevo se clasifica correctamente
• Base conceptual para algoritmos más complejos

APLICACIONES PRÁCTICAS:
• Clasificación binaria simple
• Filtrado básico de datos
• Validación de algoritmos
• Fundamento teórico esencial

CONCLUSIÓN:
Este es el escenario ideal donde los métodos lineales funcionan perfectamente, estableciendo la base para comprender casos más complejos."""
        
        self.update_info("DEMOSTRACIÓN: CASO SIMPLE", info_content)
        
    def demo_generated(self):
        """Demostración 2: Datos Generados"""
        self.current_demo = "Datos Generados"
        
        # Generar datos separables
        X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                 n_informative=2, n_clusters_per_class=1,
                                 class_sep=2.0, random_state=42)
        
        # Entrenar ambos modelos
        perceptron = Perceptron(max_iter=1000, random_state=42)
        logistic = LogisticRegression(random_state=42)
        
        perceptron.fit(X, y)
        logistic.fit(X, y)
        
        # Crear gráfico
        fig = self.create_modern_plot("Datos Generados: Comparación de Algoritmos")
        ax = fig.add_subplot(111, facecolor='#f8f9fa')
        
        # Puntos
        colors = ['#3498db', '#e74c3c']
        for i in range(2):
            mask = y == i
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=60, alpha=0.7,
                      label=f'Clase {i}', edgecolors='white', linewidth=1)
        
        # Crear malla para fronteras
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Fronteras de decisión
        Z_perc = perceptron.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z_perc = Z_perc.reshape(xx.shape)
        ax.contour(xx, yy, Z_perc, levels=[0], colors='#27ae60', linewidths=4,
                  linestyles='-', alpha=0.8)
        
        Z_log = logistic.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z_log = Z_log.reshape(xx.shape)
        ax.contour(xx, yy, Z_log, levels=[0], colors='#9b59b6', linewidths=4,
                  linestyles='--', alpha=0.8)
        
        # Leyenda personalizada
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color=colors[0], marker='o', linestyle='None', markersize=8),
                       Line2D([0], [0], color=colors[1], marker='o', linestyle='None', markersize=8),
                       Line2D([0], [0], color='#27ae60', linewidth=4),
                       Line2D([0], [0], color='#9b59b6', linewidth=4, linestyle='--')]
        
        ax.legend(custom_lines, ['Clase 0', 'Clase 1', 'Frontera Perceptrón', 'Frontera Reg. Logística'],
                 loc='upper right', framealpha=0.9)
        
        ax.set_xlabel('x₁', fontsize=12, fontweight='bold')
        ax.set_ylabel('x₂', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        self.show_plot(fig)
        
        # Calcular métricas
        acc_perc = perceptron.score(X, y)
        acc_log = logistic.score(X, y)
        
        info_content = f"""DATOS GENERADOS - ANÁLISIS COMPARATIVO

RESULTADO: Ambos algoritmos exitosos
Precisión Perceptrón: {acc_perc:.1%}
Precisión Reg. Logística: {acc_log:.1%}

CARACTERÍSTICAS DEL DATASET:
• 200 muestras sintéticas
• 2 características informativas
• Separación controlada (class_sep=2.0)
• Distribución balanceada de clases

COMPARACIÓN DE ALGORITMOS:

PERCEPTRÓN (Línea Verde):
• Algoritmo más directo
• Encuentra cualquier frontera separadora válida
• Computacionalmente eficiente
• Solución binaria (0/1)

REGRESIÓN LOGÍSTICA (Línea Púrpura):
• Enfoque probabilístico
• Optimiza la verosimilitud máxima
• Proporciona probabilidades de clase
• Más robusto estadísticamente

OBSERVACIONES CLAVE:
• Ambas fronteras son ligeramente diferentes
• Ambos modelos logran excelente rendimiento
• Las diferencias reflejan distintos criterios de optimización
• En datos perfectamente separables, ambos son efectivos

IMPLICACIONES PRÁCTICAS:
• Para clasificación simple: Perceptrón es suficiente
• Para probabilidades: Regresión Logística es mejor
• Ambos son base para métodos más avanzados"""
        
        self.update_info("DEMOSTRACIÓN: DATOS GENERADOS", info_content)
        
    def demo_xor(self):
        """Demostración 3: Problema XOR"""
        self.current_demo = "Problema XOR"
        
        # Datos XOR
        # (0,0) -> 0
        # (0,1) -> 1
        # (1,0) -> 1
        # (1,1) -> 0
        X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_xor = np.array([0, 1, 1, 0])
        
        # Crear gráfico
        fig = self.create_modern_plot("Problema XOR: Limitaciones de la Separabilidad Lineal")
        ax = fig.add_subplot(111, facecolor='#f8f9fa')
        
        # Puntos XOR con diseño especial
        ax.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], c='#3498db', s=400,
                marker='s', alpha=0.8, edgecolors='white', linewidth=4, label='Clase 0')
        ax.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='#e74c3c', s=400,
                marker='o', alpha=0.8, edgecolors='white', linewidth=4, label='Clase 1')

        # Anotaciones con estilo mejorado
        annotations = ['(0,0)→Clase 0', '(0,1)→Clase 1', '(1,0)→Clase 1', '(1,1)→Clase 0']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        offsets = [(-15, -15), (-15, 15), (15, 15), (15, -15)]
        
        for pos, ann, offset in zip(positions, annotations, offsets):
            ax.annotate(ann, pos, xytext=offset, textcoords='offset points',
                        fontsize=14, fontweight='bold', ha='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='white',
                                edgecolor='#2c3e50', linewidth=2, alpha=0.9))
        
        # Añadir un texto para indicar la imposibilidad de separación
        ax.text(0.5, -0.3, 'Imposible separar con una línea recta',
                ha='center', fontsize=16, fontweight='bold', color='#c0392b')

        # Intentar entrenar un modelo lineal para mostrar la frontera que fallará
        perceptron = Perceptron(max_iter=1000, random_state=42)
        try:
            perceptron.fit(X_xor, y_xor)
            w = perceptron.coef_[0]
            b = perceptron.intercept_[0]
            x_line = np.linspace(-0.5, 1.5, 100)
            
            # Evitar división por cero
            if w[1] != 0:
                y_line = -(w[0] * x_line + b) / w[1]
                ax.plot(x_line, y_line, color='#9b59b6', linestyle='--', linewidth=4,
                        alpha=0.8, label='Frontera del Perceptrón (inadecuada)')
            
        except Exception as e:
            # En caso de que el perceptrón no converja (como se espera)
            print(f"El Perceptrón no converge en el problema XOR: {e}")
            pass
            
        ax.set_xlabel('x₁', fontsize=12, fontweight='bold')
        ax.set_ylabel('x₂', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal', 'box')
        
        self.show_plot(fig)
        
        # Calcular métricas (serán pobres)
        logistic = LogisticRegression(random_state=42)
        logistic.fit(X_xor, y_xor)
        accuracy = logistic.score(X_xor, y_xor)
        
        info_content = f"""PROBLEMA XOR - LIMITACIÓN FUNDAMENTAL

RESULTADO: Falla predecible
Precisión: {accuracy:.1%} (≈ Azar)
Separación lineal: IMPOSIBLE

ANÁLISIS DEL PROBLEMA:
• Los puntos de la Clase 0 están en las esquinas opuestas ((0,0) y (1,1)).
• Los puntos de la Clase 1 están en las otras dos esquinas ((0,1) y (1,0)).
• No existe una sola línea recta que pueda separar estos dos conjuntos de puntos.

DEMOSTRACIÓN MATEMÁTICA:
Un clasificador lineal busca una línea de la forma $w_1x_1 + w_2x_2 + b = 0$.
Para separar el problema XOR, se deben cumplir las siguientes condiciones:
1. Puntos de Clase 0: $(0,0) \rightarrow w_1(0) + w_2(0) + b < 0 \implies b < 0$
2. Puntos de Clase 0: $(1,1) \rightarrow w_1(1) + w_2(1) + b < 0 \implies w_1 + w_2 + b < 0$
3. Puntos de Clase 1: $(0,1) \rightarrow w_1(0) + w_2(1) + b > 0 \implies w_2 + b > 0$
4. Puntos de Clase 1: $(1,0) \rightarrow w_1(1) + w_2(0) + b > 0 \implies w_1 + b > 0$

Sumando las condiciones 3 y 4: $(w_1 + b) + (w_2 + b) > 0 \implies w_1 + w_2 + 2b > 0$.
Si esto es cierto, entonces $w_1 + w_2 + b > -b$.
Pero la condición 1 nos dice que $b$ debe ser negativo, por lo que $-b$ debe ser positivo.
Esto contradice la condición 2, que exige que $w_1 + w_2 + b$ sea negativo. ¡No hay solución!

SOLUCIONES NECESARIAS:
• Transformación de Características: Mapear los datos a un espacio de mayor dimensión (p. ej., con kernels).
• Modelos No Lineales: Utilizar arquitecturas más complejas como redes neuronales con capas ocultas.
• Ejemplo Histórico: El problema XOR demostró las limitaciones del Perceptrón de una sola capa y fue un catalizador para el desarrollo de redes neuronales multicapa.

"""
        
        self.update_info("DEMOSTRACIÓN: PROBLEMA XOR", info_content)
        
    def demo_convergence(self):
        """Demostración 4: Análisis de Convergencia"""
        self.current_demo = "Análisis de Convergencia"
        
        # Datos para convergencia
        class_A = np.array([[1, 3], [2, 4], [3, 3], [4, 5], [2.5, 4.5]])
        class_B = np.array([[1, 1], [2, 1], [3, 2], [1, 2], [2.5, 1.5]])
        X = np.vstack([class_A, class_B])
        y = np.hstack([np.ones(len(class_A)), np.zeros(len(class_B))])
        
        # Simular entrenamiento paso a paso
        def simulate_perceptron_training(X, y, max_epochs=15):
            w = np.random.randn(2) * 0.1
            b = np.random.randn() * 0.1
            errors_history = []
            
            for epoch in range(max_epochs):
                errors = 0
                for i in range(len(X)):
                    prediction = 1 if (np.dot(w, X[i]) + b) > 0 else 0
                    error = y[i] - prediction
                    if error != 0:
                        w += error * X[i]
                        b += error
                        errors += 1
                errors_history.append(errors)
                if errors == 0:
                    break
            
            return errors_history
        
        errors_history = simulate_perceptron_training(X, y)
        
        # Crear gráfico
        fig = self.create_modern_plot("Análisis de Convergencia del Perceptrón")
        ax = fig.add_subplot(111, facecolor='#f8f9fa')
        
        epochs = range(len(errors_history))
        bars = ax.bar(epochs, errors_history, color='#3498db', alpha=0.8, edgecolor='white', linewidth=2)
        
        # Línea de tendencia
        ax.plot(epochs, errors_history, color='#e74c3c', marker='o', linewidth=3, markersize=8, alpha=0.8)
        
        # Anotar valores
        for i, errors in enumerate(errors_history):
            ax.annotate(f'{errors}', (i, errors), textcoords="offset points",
                       xytext=(0,10), ha='center', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Época de Entrenamiento', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Errores', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Área sombreada
        ax.fill_between(epochs, errors_history, alpha=0.3, color='#3498db')
        
        self.show_plot(fig)
        
        info_content = f"""ANÁLISIS DE CONVERGENCIA

RESULTADO: Convergencia en {len(errors_history)} épocas
TEOREMA: Convergencia garantizada para datos separables

PROCESO DE APRENDIZAJE:
1. Inicialización con pesos aleatorios
2. Presentación de cada muestra
3. Actualización solo si hay error
4. Repetición hasta convergencia

INTERPRETACIÓN DEL GRÁFICO:
• Eje X: Épocas de entrenamiento
• Eje Y: Errores por época
• Reducción progresiva hasta cero
• Convergencia = Separabilidad confirmada

CARACTERÍSTICAS CLAVE:
• Convergencia rápida indica datos bien separados
• Número finito de actualizaciones garantizado
• Independiente de inicialización de pesos
• Validación automática de separabilidad

APLICACIONES PRÁCTICAS:
• Diagnóstico de calidad de datos
• Validación de algoritmos
• Detección de problemas de entrenamiento
• Fundamento teórico para métodos avanzados"""
        
        self.update_info("DEMOSTRACIÓN: CONVERGENCIA", info_content)
        
    def demo_comparison(self):
        """Demostración 5: Comparación de Algoritmos"""
        self.current_demo = "Comparación de Algoritmos"
        
        # Generar datos para comparación
        X, y = make_classification(n_samples=150, n_features=2, n_redundant=0,
                                 n_informative=2, n_clusters_per_class=1,
                                 class_sep=1.8, random_state=42)
        
        # Entrenar ambos modelos
        perceptron = Perceptron(max_iter=1000, random_state=42)
        logistic = LogisticRegression(random_state=42)
        
        perceptron.fit(X, y)
        logistic.fit(X, y)
        
        # Crear gráfico
        fig = self.create_modern_plot("Comparación: Perceptrón vs Regresión Logística")
        ax = fig.add_subplot(111, facecolor='#f8f9fa')
        
        # Puntos
        colors = ['#3498db', '#e74c3c']
        for i in range(2):
            mask = y == i
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=80, alpha=0.7,
                      edgecolors='white', linewidth=1, label=f'Clase {i}')
        
        # Malla para fronteras
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Fronteras de decisión
        Z_perc = perceptron.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z_perc = Z_perc.reshape(xx.shape)
        ax.contour(xx, yy, Z_perc, levels=[0], colors='#27ae60', linewidths=4,
                  linestyles='-', alpha=0.9)
        
        Z_log = logistic.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z_log = Z_log.reshape(xx.shape)
        ax.contour(xx, yy, Z_log, levels=[0], colors='#9b59b6', linewidths=4,
                  linestyles='--', alpha=0.9)
        
        # Leyenda personalizada
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color=colors[0], marker='o', linestyle='None', markersize=8),
            Line2D([0], [0], color=colors[1], marker='o', linestyle='None', markersize=8),
            Line2D([0], [0], color='#27ae60', linewidth=4),
            Line2D([0], [0], color='#9b59b6', linewidth=4, linestyle='--')
        ]
        
        ax.legend(custom_lines, ['Clase 0', 'Clase 1', 'Perceptrón', 'Reg. Logística'],
                 loc='upper right', framealpha=0.9)
        
        ax.set_xlabel('x₁', fontsize=12, fontweight='bold')
        ax.set_ylabel('x₂', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        self.show_plot(fig)
        
        # Calcular métricas
        acc_perc = perceptron.score(X, y)
        acc_log = logistic.score(X, y)
        
        info_content = f"""COMPARACIÓN DE ALGORITMOS

PRECISIÓN:
• Perceptrón: {acc_perc:.3f}
• Regresión Logística: {acc_log:.3f}

DIFERENCIAS CLAVE:

PERCEPTRÓN (Línea Verde):
• Minimiza errores de clasificación
• Cualquier frontera separadora es válida
• Solución binaria (0/1)
• Computacionalmente simple
• Convergencia rápida

REGRESIÓN LOGÍSTICA (Línea Púrpura):
• Maximiza verosimilitud
• Busca frontera "más probable"
• Salidas probabilísticas
• Fundamentos estadísticos sólidos
• Más robusto a outliers

CUÁNDO USAR CADA UNO:
• Perceptrón: Clasificación simple, recursos limitados
• Reg. Logística: Necesitas probabilidades, mejor generalización

OBSERVACIONES VISUALES:
• Ambas fronteras separan correctamente
• Ligeras diferencias en orientación
• Reflejan distintos criterios de optimización
• Ambos son válidos para datos separables"""
        
        self.update_info("DEMOSTRACIÓN: COMPARACIÓN", info_content)
        
    def demo_circles(self):
        """Demostración 6: Círculos Concéntricos"""
        self.current_demo = "Círculos Concéntricos"
        
        # Generar círculos concéntricos
        np.random.seed(42)
        n_samples = 100
        r1, r2 = 0.5, 1.5
        angles = np.random.uniform(0, 2*np.pi, n_samples//2)
        
        # Círculo interno (clase 0)
        X_inner = np.column_stack([r1 * np.cos(angles), r1 * np.sin(angles)])
        X_inner += np.random.normal(0, 0.1, X_inner.shape)
        
        # Círculo externo (clase 1)
        X_outer = np.column_stack([r2 * np.cos(angles), r2 * np.sin(angles)])
        X_outer += np.random.normal(0, 0.1, X_outer.shape)
        
        X_circles = np.vstack([X_inner, X_outer])
        y_circles = np.hstack([np.zeros(len(X_inner)), np.ones(len(X_outer))])
        
        # Entrenar modelo lineal (fallará)
        logistic = LogisticRegression(random_state=42)
        logistic.fit(X_circles, y_circles)
        
        # Crear gráfico
        fig = self.create_modern_plot("Círculos Concéntricos: Caso Complejo No Separable")
        ax = fig.add_subplot(111, facecolor='#f8f9fa')
        
        # Puntos con diseño especial
        colors = ['#3498db', '#e74c3c']
        for i in range(2):
            mask = y_circles == i
            label = f'Clase {i} ({"Interno" if i == 0 else "Externo"})'
            ax.scatter(X_circles[mask, 0], X_circles[mask, 1], c=colors[i], s=100,
                      alpha=0.7, label=label, edgecolors='white', linewidth=1)
        
        # Malla para frontera lineal
        x_min, x_max = X_circles[:, 0].min() - 0.5, X_circles[:, 0].max() + 0.5
        y_min, y_max = X_circles[:, 1].min() - 0.5, X_circles[:, 1].max() + 0.5
        xx_c, yy_c = np.meshgrid(np.arange(x_min, x_max, 0.05),
                               np.arange(y_min, y_max, 0.05))
        
        # Frontera lineal inadecuada
        Z_circles = logistic.decision_function(np.c_[xx_c.ravel(), yy_c.ravel()])
        Z_circles = Z_circles.reshape(xx_c.shape)
        ax.contour(xx_c, yy_c, Z_circles, levels=[0], colors='#e67e22', linewidths=4,
                  linestyles='-', alpha=0.8, label='Frontera Lineal')
        
        # Círculos ideales de referencia
        import matplotlib.patches as patches
        circle1 = patches.Circle((0, 0), r1, fill=False, color='#27ae60',
                               linestyle='--', linewidth=3, alpha=0.7)
        circle2 = patches.Circle((0, 0), r2, fill=False, color='#27ae60',
                               linestyle='--', linewidth=3, alpha=0.7)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        
        # Leyenda personalizada
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color=colors[0], marker='o', linestyle='None', markersize=8),
            Line2D([0], [0], color=colors[1], marker='o', linestyle='None', markersize=8),
            Line2D([0], [0], color='#e67e22', linewidth=4),
            Line2D([0], [0], color='#27ae60', linewidth=3, linestyle='--')
        ]
        
        ax.legend(custom_lines, ['Clase 0 (Interno)', 'Clase 1 (Externo)', 
                                'Frontera Lineal', 'Fronteras Ideales'],
                 loc='upper right', framealpha=0.9)
        
        ax.set_xlabel('x₁', fontsize=12, fontweight='bold')
        ax.set_ylabel('x₂', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        
        # Texto explicativo
        ax.text(0, -2.2, 'Una línea recta no puede encerrar un círculo',
                ha='center', fontsize=12, fontweight='bold', color='#c0392b')
        
        self.show_plot(fig)
        
        # Calcular métricas (serán pobres)
        accuracy = logistic.score(X_circles, y_circles)
        
        info_content = f"""CÍRCULOS CONCÉNTRICOS - COMPLEJIDAD AVANZADA

RESULTADO: Falla esperada de métodos lineales
PRECISIÓN: {accuracy:.3f} (Limitada por geometría)

ESTRUCTURA DEL PROBLEMA:
• Clase 0 (azul): Círculo interno (radio ≈ 0.5)
• Clase 1 (roja): Círculo externo (radio ≈ 1.5)
• Ruido gaussiano añadido para realismo
• Patrón radial inherentemente no lineal

LIMITACIÓN GEOMÉTRICA:
Una línea recta no puede separar regiones circulares concéntricas debido a que requeriría "encerrar" completamente una región.

FRONTERA LINEAL (Naranja):
• Mejor aproximación posible
• Múltiples errores en interfaces
• Sesgo hacia una clase
• Rendimiento limitado estructuralmente

FRONTERAS IDEALES (Verde):
• Círculos concéntricos perfectos
• Separación radial completa
• Requieren métodos no lineales

SOLUCIONES NO LINEALES:
• SVM con kernel RBF
• Redes neuronales multicapa
• Random Forest y ensemble
• Transformación a coordenadas polares

APLICACIONES REALES:
• Detección de anomalías radiales
• Reconocimiento de patrones circulares
• Análisis de datos con simetría central
• Clasificación geográfica con centros"""
        
        self.update_info("DEMOSTRACIÓN: CÍRCULOS CONCÉNTRICOS", info_content)

# Función principal para ejecutar la aplicación
def main():
    """Función principal"""
    root = tk.Tk()
    app = SeparabilidadLinealApp(root)
    
    # Configurar cierre de aplicación
    def on_closing():
        if messagebox.askokcancel("Salir", "¿Deseas cerrar la aplicación?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()