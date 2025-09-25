import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BostonHousingRegressionDashboard:
    def __init__(self):
        
        self.root = tk.Tk()
        self.root.title("Dashboard 3.2 - Modelos de Regresión - Boston Housing")
        self.root.geometry("1400x800")
        self.root.configure(bg="#f0f0f0")
        
        # Variables de datos
        self.X = None
        self.y = None
        self.feature_names = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models_trained = False
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.y_pred = None
        self.scaler = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        
        self.setup_styles()
        self.load_boston_data()
        self.setup_dashboard()
    
    def setup_styles(self):
        """Configurar estilos de widgets"""
        style = ttk.Style(self.root)
        style.theme_use('clam')
        
        # Estilo para botones principales
        style.configure('Main.TButton',
                        font=("Helvetica", 12, "bold"),
                        background="#007acc",
                        foreground="#ffffff",
                        padding=12)
        style.map('Main.TButton',
                  background=[('active', '#005f99')])

        # Estilo para botones de navegación
        style.configure('Nav.TButton',
                        font=("Arial", 10),
                        background="#e0e0e0",
                        foreground="#333333",
                        padding=8)
        style.map('Nav.TButton',
                  background=[('active', '#cccccc')])
    
    def load_boston_data(self):
        """Cargar datos del dataset de Boston Housing desde CSV"""
        try:
            # Primero intentar cargar archivo CSV si existe
            # Si no, usar datos predeterminados
            self.create_default_dataset()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error cargando datos: {e}")
            self.root.quit()
    
    def load_csv_file(self):
        """Cargar archivo CSV del usuario - VERSIÓN MEJORADA"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo CSV de Boston Housing",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Leer el archivo línea por línea y procesar los datos
                data = []
                with open(file_path, 'r') as file:
                    for line in file:
                        # Limpiar espacios múltiples y dividir por espacios
                        cleaned_line = ' '.join(line.split())
                        values = cleaned_line.split()
                        
                        # Convertir a números flotantes
                        try:
                            numeric_values = [float(val) for val in values]
                            data.append(numeric_values)
                        except ValueError:
                            continue
                
                # Verificar que tenemos datos válidos
                if len(data) == 0:
                    messagebox.showerror("Error", "El archivo no contiene datos válidos o el formato es incorrecto")
                    return
                
                # Verificar que todas las filas tienen la misma longitud
                row_lengths = [len(row) for row in data]
                if min(row_lengths) != max(row_lengths):
                    messagebox.showerror("Error", "Las filas del archivo tienen longitudes inconsistentes")
                    return
                
                # Nombres de las columnas según el dataset de vivienda de Boston
                column_names = [
                    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 
                    'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
                ]
                
                # Verificar que tenemos el número correcto de columnas
                if len(data[0]) != len(column_names):
                    messagebox.showerror("Error", f"El archivo debe tener exactamente {len(column_names)} columnas. Encontradas: {len(data[0])}")
                    return
                
                # Crear DataFrame
                self.df = pd.DataFrame(data, columns=column_names)
                self.y = self.df['MEDV'].values
                self.X = self.df.drop('MEDV', axis=1).values
                self.feature_names = self.df.columns[:-1].tolist()
                
                self.prepare_data()
                messagebox.showinfo("Éxito", f"Dataset cargado: {len(self.df)} muestras, {len(self.feature_names)} características")
                
                # Actualizar la interfaz
                self.update_display()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error leyendo archivo: {e}\n\nAsegúrese de que el archivo tenga el formato correcto:\n- 13 características + 1 variable objetivo\n- Datos numéricos\n- Separados por espacios")
    
    def create_default_dataset(self):
        """Crear dataset por defecto basado en los datos proporcionados"""
        # Datos del CSV proporcionado por el usuario
        data = [
            [0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90, 4.98, 24.00],
            [0.02731, 0.00, 7.070, 0, 0.4690, 6.4210, 78.90, 4.9671, 2, 242.0, 17.80, 396.90, 9.14, 21.60],
            [0.02729, 0.00, 7.070, 0, 0.4690, 7.1850, 61.10, 4.9671, 2, 242.0, 17.80, 392.83, 4.03, 34.70],
            [0.03237, 0.00, 2.180, 0, 0.4580, 6.9980, 45.80, 6.0622, 3, 222.0, 18.70, 394.63, 2.94, 33.40],
            [0.06905, 0.00, 2.180, 0, 0.4580, 7.1470, 54.20, 6.0622, 3, 222.0, 18.70, 396.90, 5.33, 36.20]
        ]
        
        # Nombres de las características
        self.feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
                             'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        
        # Crear DataFrame
        self.df = pd.DataFrame(data, columns=self.feature_names + ['MEDV'])
        self.X = self.df[self.feature_names].values
        self.y = self.df['MEDV'].values
        
        self.prepare_data()
    
    def prepare_data(self):
        """Preparar datos para el entrenamiento"""
        # Dividir datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=42)
        
        # Escalar datos
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def update_display(self):
        """Actualizar la visualización después de cargar nuevos datos"""
        # Actualizar el texto de bienvenida
        for widget in self.plot_frame.winfo_children():
            if isinstance(widget, ttk.Frame) and hasattr(widget, 'welcome_label'):
                widget.welcome_label.config(
                    text=f"Dataset: Boston Housing\nMuestras: {len(self.df)}\nCaracterísticas: {len(self.feature_names)}\nObjetivo: MEDV (Precio vivienda)"
                )
    
    def setup_dashboard(self):
        """Configurar dashboard principal"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título y botón de carga
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(title_frame, text="3.2 - Modelos de Regresión - Boston Housing", 
                 font=("Helvetica", 18, "bold")).pack(side=tk.LEFT)
        
        ttk.Button(title_frame, text="Cargar CSV", command=self.load_csv_file,
                  style="Main.TButton").pack(side=tk.RIGHT, padx=(10, 0))
        
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo - Botones
        left_panel = ttk.LabelFrame(content_frame, text="Análisis Disponibles", padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # Panel central - Visualización
        center_panel = ttk.LabelFrame(content_frame, text="Visualización", padding="10")
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Panel derecho - Explicación
        right_panel = ttk.LabelFrame(content_frame, text="Interpretación y Análisis", padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        self.setup_buttons(left_panel)
        self.setup_plot_area(center_panel)
        self.setup_explanation_area(right_panel)
    
    def setup_buttons(self, parent):
        """Configurar botones de análisis"""
        analyses = [
            ("Información Dataset", "Estadísticas descriptivas", self.show_dataset_info),
            ("Distribución Objetivo", "Histograma MEDV", self.plot_target_distribution),
            ("Matriz Correlación", "Correlaciones entre variables", self.plot_correlation_matrix),
            ("Habitaciones vs Precio", "Análisis RM vs MEDV", self.plot_rooms_vs_price),
            ("LSTAT vs Precio", "Nivel socioeconómico", self.plot_lstat_vs_price),
            ("Modelos Base", "Linear y Ridge", self.train_baseline_models),
            ("Comparación Avanzada", "Todos los modelos", self.compare_all_models),
            ("Predicciones Finales", "Mejor modelo", self.plot_final_predictions)
        ]
        
        for name, desc, func in analyses:
            btn = ttk.Button(parent, text=name, width=25, style="Nav.TButton",
                           command=lambda n=name, d=desc, f=func: self.show_analysis(n, d, f))
            btn.pack(pady=3, fill=tk.X)
        
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Botón para cerrar
        ttk.Button(parent, text="Cerrar Aplicación", command=self.root.quit,
                  style="Main.TButton").pack(pady=10, fill=tk.X)
    
    def setup_plot_area(self, parent):
        """Configurar área de visualización"""
        self.plot_frame = parent
        
        welcome_frame = ttk.Frame(parent)
        welcome_frame.pack(expand=True, fill=tk.BOTH)
        welcome_frame.welcome_label = ttk.Label(welcome_frame, 
                 text=f"Dataset: Boston Housing\nMuestras: {len(self.df) if self.df is not None else 'N/A'}\nCaracterísticas: {len(self.feature_names) if self.feature_names is not None else 'N/A'}\nObjetivo: MEDV (Precio vivienda)",
                 font=("Arial", 12), justify=tk.CENTER)
        welcome_frame.welcome_label.pack(pady=10)
        
        ttk.Label(welcome_frame, text="Selecciona un análisis del panel izquierdo",
                 font=("Arial", 11), foreground="gray").pack(pady=10)
    
    def setup_explanation_area(self, parent):
        """Configurar área de explicaciones"""
        self.explanation_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, 
                                                         width=45, height=35, font=("Arial", 10))
        self.explanation_text.pack(fill=tk.BOTH, expand=True)
        
        self.explanation_text.insert(tk.END, 
                                    "ANÁLISIS DE REGRESIÓN - BOSTON HOUSING\n\n"
                                    "OBJETIVO DEL EJERCICIO:\n"
                                    "Predecir el valor medio de viviendas residenciales usando variables socioeconómicas, demográficas y urbanas.\n\n"
                                    "METODOLOGÍA:\n"
                                    "1. Exploración de datos\n"
                                    "2. Análisis de correlaciones\n"
                                    "3. Entrenamiento de modelos\n"
                                    "4. Validación cruzada\n"
                                    "5. Evaluación final\n\n"
                                    "MÉTRICAS PRINCIPALES:\n"
                                    "• MAE: Error Absoluto Medio\n"
                                    "• RMSE: Raíz del Error Cuadrático Medio\n"
                                    "• R²: Coeficiente de Determinación\n\n"
                                    "Selecciona un análisis para comenzar...")
    
    def show_analysis(self, name, description, analysis_function):
        """Mostrar análisis seleccionado"""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        self.update_explanation(name)
        analysis_function()
    
    def update_explanation(self, analysis_name):
        """Actualizar explicación según el análisis"""
        explanations = {
            "Información Dataset": """ESTADÍSTICAS DESCRIPTIVAS DEL DATASET

PROPÓSITO:
Proporcionar una vista general de las características del dataset Boston Housing, incluyendo medidas de tendencia central y dispersión.

VARIABLES DEL DATASET:
• CRIM: Tasa de criminalidad per cápita
• ZN: Proporción de terrenos zonificados
• INDUS: Proporción de acres comerciales no minoristas
• CHAS: Variable ficticia río Charles (1 si limita, 0 no)
• NOX: Concentración de óxidos nítricos
• RM: Número promedio de habitaciones por vivienda
• AGE: Proporción de unidades ocupadas por propietarios construidas antes de 1940
• DIS: Distancias ponderadas a centros de empleo
• RAD: Índice de accesibilidad a carreteras radiales
• TAX: Tasa de impuesto a la propiedad por $10,000
• PTRATIO: Ratio alumno-profesor por ciudad
• B: Proporción de afroamericanos por ciudad
• LSTAT: % población de estatus socioeconómico bajo
• MEDV: Valor medio de vivienda (OBJETIVO)

INTERPRETACIÓN:
Las estadísticas nos ayudan a:
1. Identificar outliers potenciales
2. Entender las escalas de las variables
3. Detectar necesidad de transformaciones
4. Planificar estrategias de preprocesamiento

VARIABLES CLAVE IDENTIFICADAS:
RM (habitaciones) y LSTAT (estatus socioeconómico) típicamente muestran las correlaciones más fuertes con el precio.""",

            "Distribución Objetivo": """DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (MEDV)

ANÁLISIS:
Este histograma muestra la distribución del valor medio de las viviendas en el dataset.

ELEMENTOS VISUALES:
• Eje X: Valor de la vivienda (en miles de USD)
• Eje Y: Frecuencia (número de observaciones)
• Línea roja: Media del dataset
• Curva: Estimación de densidad

PATRONES IDENTIFICADOS:
Distribución: La mayoría de valores se concentran entre $15k-$25k
Sesgo: Distribución sesgada a la derecha (cola larga hacia valores altos)
Pico artificial: Concentración en $50k debido a censoring en datos originales
Rango: Valores típicos entre $5k - $50k

IMPLICACIONES PARA EL MODELO:
1. El sesgo puede requerir transformación logarítmica
2. Los valores censurados en $50k pueden afectar predicciones altas
3. La mayoría de datos están en rango medio-bajo
4. Outliers en el extremo superior requieren atención especial

IMPORTANCIA:
Entender la distribución del objetivo es crucial para:
• Seleccionar métricas de evaluación apropiadas
• Decidir sobre transformaciones
• Interpretar errores del modelo correctamente""",

            "Matriz Correlación": """MATRIZ DE CORRELACIONES

INTERPRETACIÓN DEL MAPA DE CALOR:
Rojo intenso: Correlación positiva fuerte (cercana a +1)
Azul intenso: Correlación negativa fuerte (cercana a -1)
Blanco: Sin correlación (cercana a 0)

CORRELACIONES CLAVE CON MEDV (PRECIO):
POSITIVAS FUERTES:
• RM (habitaciones): ~0.7 - Más habitaciones = Mayor precio
• ZN (zonificación): ~0.4 - Mejor zonificación = Mayor precio

NEGATIVAS FUERTES:
• LSTAT (estatus bajo): ~-0.7 - Mayor pobreza = Menor precio  
• PTRATIO (ratio estudiante-profesor): ~-0.5 - Peores escuelas = Menor precio
• CRIM (criminalidad): ~-0.4 - Más crimen = Menor precio

MULTICOLINEALIDAD DETECTADA:
RAD-TAX: Correlación muy alta (~0.9)
NOX-DIS: Correlación negativa fuerte
AGE-NOX: Correlación positiva moderada

INSIGHTS PARA EL MODELO:
1. RM y LSTAT son predictores dominantes
2. Posible multicolinealidad requiere regularización
3. Algunas variables pueden ser redundantes
4. Variables geográficas (DIS, RAD) muestran patrones interesantes

ESTRATEGIAS:
• Usar Ridge/Lasso para manejar multicolinealidad
• Considerar selección de características
• Monitorear variables altamente correlacionadas""",

            "Habitaciones vs Precio": """ANÁLISIS: HABITACIONES vs PRECIO

RELACIÓN PRINCIPAL:
Esta es típicamente la relación más fuerte en datasets inmobiliarios.

PATRONES OBSERVADOS:
Tendencia: Clara relación positiva y casi lineal
Dispersión: Aumenta ligeramente con más habitaciones
Rango típico: 4-8 habitaciones por vivienda
Outliers: Algunas viviendas con 8+ habitaciones

ANÁLISIS ESTADÍSTICO:
• Correlación: ~0.70 (muy fuerte)
• Relación: Aproximadamente lineal
• Pendiente: ~$9k por habitación adicional
• Variabilidad: Mayor en viviendas grandes

INTERPRETACIÓN PRÁCTICA:
1. Cada habitación adicional añade valor significativo
2. La relación es consistente en todo el rango
3. Viviendas con 6+ habitaciones en mercado premium
4. Pocos outliers - relación robusta

IMPLICACIONES PARA EL MODELO:
RM será predictor dominante
Relación lineal favorable para regresión lineal
Baja variabilidad residual
Atención a outliers en extremos

VALOR PREDICTIVO:
Solo con RM se puede explicar ~49% de la variabilidad en precios, confirmando su importancia como característica principal.""",

            "LSTAT vs Precio": """ANÁLISIS: NIVEL SOCIOECONÓMICO vs PRECIO

VARIABLE LSTAT:
Porcentaje de población con estatus socioeconómico bajo en el área.

RELACIÓN OBSERVADADA:
Correlación: Fuertemente negativa (~-0.74)
Forma: Relación no-lineal (exponencial/cuadrática)
Patrón: Decaimiento más pronunciado en valores bajos de LSTAT

INTERPRETACIÓN SOCIOECONÓMICA:
• LSTAT bajo (0-10%): Áreas prósperas, precios altos
• LSTAT medio (10-20%): Áreas mixtas, precios moderados  
• LSTAT alto (20%+): Áreas desfavorecidas, precios bajos

CARACTERÍSTICAS DE LA RELACIÓN:
1. No es perfectamente lineal
2. Mayor dispersión en valores medios de LSTAT
3. Relación más fuerte que muchas variables económicas
4. Efecto de umbral: cambios dramáticos en ciertos rangos

IMPLICACIONES PARA MODELADO:
La no-linealidad sugiere que modelos como Random Forest pueden capturar mejor esta relación
Posible transformación polinomial para regresión lineal
Variable clave para segmentación de mercado

CONTEXTO URBANO:
Esta relación refleja patrones socioeconómicos urbanos donde la composición social del vecindario influye fuertemente en los valores inmobiliarios.""",

            "Modelos Base": """MODELOS BASELINE - REGRESIÓN LINEAL vs RIDGE

MODELOS IMPLEMENTADOS:

1. REGRESIÓN LINEAL:
• Algoritmo: Mínimos cuadrados ordinarios
• Supuestos: Relación lineal, independencia, homocedasticidad
• Ventajas: Interpretabilidad, simplicidad, rápido
• Desventajas: Sensible a multicolinealidad

2. RIDGE REGRESSION:
• Algoritmo: Mínimos cuadrados con regularización L2
• Parámetro: Alpha = 1.0
• Ventajas: Maneja multicolinealidad, reduces overfitting
• Desventajas: Coeficientes sesgados, menos interpretable

MÉTRICAS DE EVALUACIÓN:
MAE (Error Absoluto Medio): Promedio errores absolutos
RMSE (Raíz Error Cuadrático Medio): Penaliza errores grandes
R² (Coeficiente Determinación): % varianza explicada

COMPARACIÓN ESPERADA:
• Ambos modelos deberían mostrar R² > 0.6
• Ridge típicamente con RMSE ligeramente menor
• Diferencias pequeñas indican dataset bien comportado
• MAE típicamente entre $3-5k

VALIDACIÓN:
• División 75% entrenamiento, 25% prueba
• Datos estandarizados para Ridge
• Validación cruzada para robustez

INTERPRETACIÓN DE RESULTADOS:
Un R² > 0.65 indica que las variables lineales explican bien los precios de vivienda.""",

            "Comparación Avanzada": """COMPARACIÓN COMPLETA DE MODELOS

MODELOS EVALUADOS:

1. REGRESIÓN LINEAL (Baseline):
• Interpretación directa de coeficientes
• Asume relaciones lineales puras

2. RIDGE REGRESSION:
• Regularización L2 para multicolinealidad
• Reduce varianza del modelo

3. RANDOM FOREST:
• Ensemble de árboles de decisión
• Captura interacciones no-lineales
• Robusto a outliers y valores faltantes

VALIDACIÓN CRUZADA:
• K-Fold con 5 particiones
• Métricas promediadas para mayor confiabilidad
• Desviación estándar para medir consistencia

MÉTRICAS COMPARATIVAS:
R² Score: Capacidad explicativa
MAE: Error promedio en dólares
RMSE: Sensibilidad a errores grandes

RESULTADOS ESPERADOS:
Random Forest: Mejor R² (~0.85-0.90)
Ridge: R² moderado (~0.70-0.75)
Linear: R² baseline (~0.65-0.70)

SELECCIÓN DEL MEJOR MODELO:
• Criterio principal: Mayor R² en validación cruzada
• Criterio secundario: Menor MAE para interpretabilidad
• Consideración: Balance entre precisión y complejidad

ANÁLISIS DE RESULTADOS:
Si Random Forest supera significativamente a los modelos lineales, confirma la presencia de relaciones no-lineales importantes en el dataset.""",

            "Predicciones Finales": """ANÁLISIS FINAL - PREDICCIONES vs VALORES REALES

GRÁFICO DE DISPERSIÓN:
• Eje X: Valores reales (y_test)
• Eje Y: Predicciones del mejor modelo (y_pred)
• Línea roja diagonal: Predicción perfecta (45°)

INTERPRETACIÓN VISUAL:
Puntos cerca de la línea: Predicciones precisas
Puntos alejados: Errores significativos
Patrón de dispersión: Indica calidad del modelo

ANÁLISIS DE ERRORES:
Errores sistemáticos: Sesgo del modelo
Heteroscedasticidad: Variabilidad no constante
Outliers: Casos problemáticos específicos

MÉTRICAS FINALES:
• MAE: Error promedio en términos monetarios
• RMSE: Sensibilidad a errores grandes  
• R²: Porcentaje de varianza explicada

REGIONES DE ANÁLISIS:
1. Precios bajos ($5-20k): ¿Subestimación sistemática?
2. Precios medios ($20-35k): ¿Mayor precisión?
3. Precios altos ($35-50k): ¿Problemas con valores censurados?

CONCLUSIONES DEL MODELO:
• Capacidad predictiva para planificación urbana
• Limitaciones en valores extremos
• Utilidad para evaluación inmobiliaria automatizada
• Factores no capturados que requieren investigación adicional

RECOMENDACIONES:
Para uso práctico, considerar intervalos de confianza y validación continua con datos nuevos."""
        }
        
        self.explanation_text.delete(1.0, tk.END)
        self.explanation_text.insert(tk.END, explanations.get(analysis_name, "Explicación no disponible"))
    
    def show_dataset_info(self):
        """Mostrar información descriptiva del dataset"""
        # Crear tabla con datos del dataset
        table_frame = ttk.Frame(self.plot_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Crear Treeview para mostrar los datos
        columns = list(self.df.columns)
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=25)
        
        # Configurar columnas
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor=tk.CENTER)
        
        # Agregar scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Añadir TODOS los datos a la tabla
        for i, row in self.df.iterrows():
            tree.insert('', tk.END, values=list(row))
        
        # Empaquetar widgets
        tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        # Configurar grid weights
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Estadísticas descriptivas
        stats_text = f"Número de muestras: {len(self.df)}\n"
        stats_text += f"Número de características: {len(self.feature_names)}\n"
        stats_text += f"Variable objetivo: MEDV (Valor medio vivienda)\n\n"
        stats_text += f"Precio promedio: ${self.df['MEDV'].mean():.1f}k\n"
        stats_text += f"Precio mediano: ${self.df['MEDV'].median():.1f}k\n"
        stats_text += f"Rango de precios: ${self.df['MEDV'].min():.1f}k - ${self.df['MEDV'].max():.1f}k\n"
        stats_text += f"Habitaciones promedio: {self.df['RM'].mean():.1f}\n"
        stats_text += f"Criminalidad promedio: {self.df['CRIM'].mean():.2f}\n"
        
        stats_label = ttk.Label(self.plot_frame, text=stats_text, font=("Arial", 10))
        stats_label.pack(pady=10)
    
    def plot_target_distribution(self):
        """Histograma de la variable objetivo"""
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Histograma con curva de densidad
        n_bins = min(30, len(self.df) // 5)
        ax.hist(self.df['MEDV'], bins=n_bins, density=True, alpha=0.7, 
                color='skyblue', edgecolor='black', linewidth=0.5)
        
        # Línea de la media
        mean_val = self.df['MEDV'].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Media: ${mean_val:.1f}k')
        
        # Estadísticas adicionales
        median_val = self.df['MEDV'].median()
        ax.axvline(median_val, color='green', linestyle=':', linewidth=2,
                   label=f'Mediana: ${median_val:.1f}k')
        
        ax.set_title('Distribución del Valor de las Viviendas (MEDV)', 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Valor de la Vivienda (miles de USD)', fontsize=12)
        ax.set_ylabel('Densidad', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Agregar texto con estadísticas
        stats_text = f'Desviación Estándar: ${self.df["MEDV"].std():.1f}k\n'
        stats_text += f'Rango: ${self.df["MEDV"].min():.1f}k - ${self.df["MEDV"].max():.1f}k\n'
        stats_text += f'Coeficiente Variación: {(self.df["MEDV"].std()/self.df["MEDV"].mean())*100:.1f}%'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_correlation_matrix(self):
        """Mapa de calor de la matriz de correlación"""
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Calcular matriz de correlación
        corr_matrix = self.df.corr()
        
        # Crear mapa de calor
        im = ax.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Configurar etiquetas
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        # Agregar valores de correlación
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white",
                              fontweight='bold' if abs(corr_matrix.iloc[i, j]) > 0.7 else 'normal')
        
        ax.set_title('Matriz de Correlación - Boston Housing Dataset', 
                     fontsize=16, fontweight='bold')
        
        # Barra de colores
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Correlación', rotation=270, labelpad=20)
        
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mostrar correlaciones más importantes con MEDV
        medv_corr = corr_matrix['MEDV'].sort_values(key=abs, ascending=False)[1:]
        top_corr_text = "Top 5 Correlaciones con MEDV:\n"
        for i, (var, corr) in enumerate(medv_corr.head().items()):
            top_corr_text += f"{i+1}. {var}: {corr:.3f}\n"
        
        info_label = ttk.Label(self.plot_frame, text=top_corr_text, 
                              font=("Arial", 10, "bold"), foreground="darkblue")
        info_label.pack(pady=5)
    
    def plot_rooms_vs_price(self):
        """Gráfico de dispersión mejorado: Habitaciones vs Precio"""
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # MEJORA 1: Categorizar precios y usar colores intuitivos como en tu imagen
        # Crear categorías de precio
        price_low = self.df['MEDV'].quantile(0.33)
        price_high = self.df['MEDV'].quantile(0.67)
        
        # Definir colores para cada categoría (como en tu imagen)
        colors = []
        price_categories = []
        for price in self.df['MEDV']:
            if price <= price_low:
                colors.append('#5975A4')  # Azul para precios bajos
                price_categories.append('low')
            elif price <= price_high:
                colors.append('#5F9E6E')  # Verde para precios medios  
                price_categories.append('med')
            else:
                colors.append('#CC8963')  # Naranja/salmón para precios altos
                price_categories.append('high')
        
        # Scatter plot con colores por categorías
        scatter = ax.scatter(self.df['RM'], self.df['MEDV'], 
                           alpha=0.7, s=60, 
                           c=colors,
                           edgecolors='black', 
                           linewidth=0.5)
        
        # MEJORA 2: Línea de tendencia más visible
        z = np.polyfit(self.df['RM'], self.df['MEDV'], 1)
        p = np.poly1d(z)
        ax.plot(self.df['RM'], p(self.df['RM']), 
               color='red', linestyle='-', linewidth=3, alpha=0.8,
               label=f'Tendencia: y = {z[0]:.1f}x + {z[1]:.1f}')
        
        # MEJORA 3: Títulos y etiquetas más claros
        ax.set_title('Número de Habitaciones vs Precio de Vivienda\n(Relación Positiva Fuerte)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Número promedio de habitaciones (RM)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precio vivienda - MEDV (miles USD)', fontsize=12, fontweight='bold')
        
        # MEJORA 4: Grid más sutil
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # MEJORA 5: Leyenda de colores como en tu imagen
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#5975A4', label=f'Bajo (≤${price_low:.1f}k)'),
            Patch(facecolor='#5F9E6E', label=f'Medio (${price_low:.1f}k-${price_high:.1f}k)'),
            Patch(facecolor='#CC8963', label=f'Alto (≥${price_high:.1f}k)'),
            ax.lines[0]  # Línea de tendencia
        ]
        ax.legend(handles=legend_elements, fontsize=10, loc='lower right', title='PRICE_CAT')
        
        # MEJORA 6: Estadísticas más prominentes
        correlation = self.df['RM'].corr(self.df['MEDV'])
        r_squared = correlation ** 2
        
        # Crear caja de estadísticas más llamativa
        stats_text = f'ESTADÍSTICAS:\n'
        stats_text += f'Correlación: {correlation:.3f}\n'
        stats_text += f'R² explicado: {r_squared*100:.1f}%\n'
        stats_text += f'Relación: MUY FUERTE'
        
        # Caja de estadísticas con mejor contraste
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', 
                         edgecolor='darkgreen', linewidth=2, alpha=0.9))
        
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_lstat_vs_price(self):
        """Gráfico de dispersión mejorado: LSTAT vs Precio"""
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # MEJORA 1: Usar el mismo esquema de colores por categorías de precio
        price_low = self.df['MEDV'].quantile(0.33)
        price_high = self.df['MEDV'].quantile(0.67)
        
        # Definir colores consistentes con la gráfica de habitaciones
        colors = []
        price_categories = []
        for price in self.df['MEDV']:
            if price <= price_low:
                colors.append('#5975A4')  # Azul para precios bajos
                price_categories.append('low')
            elif price <= price_high:
                colors.append('#5F9E6E')  # Verde para precios medios  
                price_categories.append('med')
            else:
                colors.append('#CC8963')  # Naranja/salmón para precios altos
                price_categories.append('high')
        
        # Scatter plot con colores por categorías
        scatter = ax.scatter(self.df['LSTAT'], self.df['MEDV'], 
                           alpha=0.7, s=60,
                           c=colors,
                           edgecolors='black', 
                           linewidth=0.5)
        
        # MEJORA 2: Línea de tendencia más clara con curva polinomial
        z = np.polyfit(self.df['LSTAT'], self.df['MEDV'], 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(self.df['LSTAT'].min(), self.df['LSTAT'].max(), 100)
        ax.plot(x_smooth, p(x_smooth), 
               color='darkred', linestyle='-', linewidth=3, alpha=0.9,
               label='Tendencia cuadrática')
        
        # MEJORA 3: Títulos más descriptivos
        ax.set_title('Nivel Socioeconómico vs Precio de Vivienda\n(Relación Negativa No-Lineal)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('% Población de bajo estatus socioeconómico (LSTAT)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precio vivienda - MEDV (miles USD)', fontsize=12, fontweight='bold')
        
        # MEJORA 4: Grid y leyenda consistente
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Leyenda de colores consistente con la otra gráfica
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#5975A4', label=f'Bajo (≤${price_low:.1f}k)'),
            Patch(facecolor='#5F9E6E', label=f'Medio (${price_low:.1f}k-${price_high:.1f}k)'),
            Patch(facecolor='#CC8963', label=f'Alto (≥${price_high:.1f}k)'),
            ax.lines[0]  # Línea de tendencia
        ]
        ax.legend(handles=legend_elements, fontsize=10, loc='upper right', title='PRICE_CAT')
        
        # MEJORA 5: Estadísticas mejoradas
        correlation = self.df['LSTAT'].corr(self.df['MEDV'])
        
        stats_text = f'ANÁLISIS SOCIOECONÓMICO:\n'
        stats_text += f'Correlación: {correlation:.3f}\n'
        stats_text += f'Tipo: FUERTEMENTE NEGATIVA\n'
        stats_text += f'Forma: EXPONENCIAL\n'
        stats_text += f'A mayor pobreza, menor precio'
        
        # Caja de estadísticas con colores que reflejen la relación negativa
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                         edgecolor='orange', linewidth=2, alpha=0.9))
        
        # MEJORA 6: Marcadores de zonas socioeconómicas más sutiles
        # Agregar líneas verticales para delimitar zonas
        ax.axvline(10, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
        ax.axvline(20, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
        
        # Etiquetas para las zonas más discretas
        ax.text(5, ax.get_ylim()[1]*0.95, 'Zona\nPróspera', ha='center', 
               fontweight='bold', color='darkgreen', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
        ax.text(15, ax.get_ylim()[1]*0.95, 'Zona\nMixta', ha='center', 
               fontweight='bold', color='darkorange', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
        ax.text(30, ax.get_ylim()[1]*0.95, 'Zona\nDesfavorecida', ha='center', 
               fontweight='bold', color='darkred', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))
        
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def train_baseline_models(self):
        """Entrenar modelos baseline y mostrar resultados"""
        # Verificar que hay suficientes datos para entrenamiento
        if len(self.X_train) < 2:
            messagebox.showerror("Error", "No hay suficientes datos para entrenar los modelos. Cargue un CSV con más muestras.")
            return
            
        # Modelos baseline
        models = {
            "Regresión Lineal": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0)
        }
        
        results_text = "RESULTADOS DE MODELOS BASELINE\n"
        results_text += "=" * 50 + "\n\n"
        
        baseline_results = {}
        
        for name, model in models.items():
            # Entrenar modelo
            if name == "Ridge Regression":
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            # Calcular métricas
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            
            # Validación cruzada
            try:
                if name == "Ridge Regression":
                    cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=min(5, len(self.X_train)), scoring='r2')
                else:
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=min(5, len(self.X_train)), scoring='r2')
                
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = np.nan
                cv_std = np.nan
            
            baseline_results[name] = {
                'MAE': mae, 'RMSE': rmse, 'R2': r2, 
                'CV_mean': cv_mean, 'CV_std': cv_std
            }
            
            results_text += f"{name}:\n"
            results_text += f"  MAE (Error Absoluto Medio): ${mae:.2f}k\n"
            results_text += f"  RMSE (Raíz Error Cuadrático): ${rmse:.2f}k\n"
            results_text += f"  R² (Coef. Determinación): {r2:.4f}\n"
            if not np.isnan(cv_mean):
                results_text += f"  Validación Cruzada R²: {cv_mean:.4f} ± {cv_std:.4f}\n\n"
            else:
                results_text += f"  Validación Cruzada: No disponible (datos insuficientes)\n\n"
        
        # Comparación
        results_text += "COMPARACIÓN:\n"
        results_text += "-" * 20 + "\n"
        
        best_model = max(baseline_results.items(), key=lambda x: x[1]['R2'])
        results_text += f"Mejor modelo: {best_model[0]}\n"
        results_text += f"Diferencia R²: {abs(baseline_results['Regresión Lineal']['R2'] - baseline_results['Ridge Regression']['R2']):.4f}\n\n"
        
        results_text += "INTERPRETACIÓN:\n"
        results_text += "-" * 15 + "\n"
        results_text += f"• R² > 0.6 indica buen ajuste lineal\n"
        results_text += f"• MAE representa error promedio en términos monetarios\n"
        results_text += f"• Ridge ayuda con multicolinealidad si hay mejora significativa\n"
        
        # Guardar resultados para uso posterior
        self.baseline_results = baseline_results
        
        # Mostrar resultados
        text_widget = scrolledtext.ScrolledText(self.plot_frame, wrap=tk.WORD, 
                                                width=90, height=35, font=("Courier", 10))
        text_widget.insert(tk.END, results_text)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def compare_all_models(self):
        """Comparar todos los modelos incluyendo Random Forest"""
        if not self.models_trained:
            self.train_all_models()
        
        # Crear gráfico de barras comparativo
        fig = Figure(figsize=(6,4), dpi=100)
        
        # Subplot para R²
        ax1 = fig.add_subplot(121)
        models = list(self.results.keys())
        r2_scores = [self.results[model]['R2'] for model in models]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax1.bar(models, r2_scores, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Comparación R² - M. Regresión', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Coeficiente de Determinación (R²)', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Agregar valores en las barras
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Subplot para MAE
        ax2 = fig.add_subplot(122)
        mae_scores = [self.results[model]['MAE'] for model in models]
        
        bars2 = ax2.bar(models, mae_scores, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Comparación MAE - E Absoluto M', fontsize=14, fontweight='bold')
        ax2.set_ylabel('MAE (miles USD)', fontsize=12)
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Agregar valores en las barras
        for bar, score in zip(bars2, mae_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{score:.2f}k', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mostrar métricas detalladas
        metrics_text = f"MEJOR MODELO: {self.best_model_name}\n"
        metrics_text += f"R²: {self.results[self.best_model_name]['R2']:.4f} | "
        metrics_text += f"MAE: ${self.results[self.best_model_name]['MAE']:.2f}k | "
        metrics_text += f"RMSE: ${self.results[self.best_model_name]['RMSE']:.2f}k"
        
        info_label = ttk.Label(self.plot_frame, text=metrics_text,
                              font=("Arial", 11, "bold"), foreground="darkgreen")
        info_label.pack(pady=10)
    
    def train_all_models(self):
        """Entrenar todos los modelos"""
        # Verificar que hay suficientes datos para entrenamiento
        if len(self.X_train) < 2:
            messagebox.showerror("Error", "No hay suficientes datos para entrenar los modelos. Cargue un CSV con más muestras.")
            return
            
        models = {
            "R. Lineal": LinearRegression(),
            "Ridge R.": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        }
        
        self.results = {}
        best_r2 = -np.inf
        
        for name, model in models.items():
            if name in ["Ridge R."]:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=min(5, len(self.X_train)), scoring='r2')
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=min(5, len(self.X_train)), scoring='r2')
            
            # Calcular métricas
            r2 = r2_score(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            self.results[name] = {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "CV_R2_mean": cv_scores.mean(),
                "CV_R2_std": cv_scores.std(),
                "Model": model,
                "Predictions": y_pred
            }
            
            # Encontrar mejor modelo
            if r2 > best_r2:
                best_r2 = r2
                self.best_model = model
                self.best_model_name = name
                self.y_pred = y_pred
        
        self.models_trained = True
    
    def plot_final_predictions(self):
        """Gráfico final de predicciones vs valores reales"""
        if not self.models_trained:
            self.train_all_models()
        
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Scatter plot predicciones vs reales
        scatter = ax.scatter(self.y_test, self.y_pred, alpha=0.6, s=60, 
                            c=abs(self.y_test - self.y_pred), cmap='Reds', 
                            edgecolors='black', linewidth=0.5)
        
        # Línea de predicción perfecta
        min_val = min(self.y_test.min(), self.y_pred.min())
        max_val = max(self.y_test.max(), self.y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Predicción Perfecta', alpha=0.8)
        
        # Líneas de error ±10%
        ax.plot([min_val, max_val], [min_val*0.9, max_val*0.9], 'g:', alpha=0.6, label='±10% Error')
        ax.plot([min_val, max_val], [min_val*1.1, max_val*1.1], 'g:', alpha=0.6)
        
        ax.set_title(f'Predicciones vs Valores Reales - {self.best_model_name}', 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Valores Reales MEDV (miles USD)', fontsize=12)
        ax.set_ylabel('Predicciones MEDV (miles USD)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Barra de colores para errores
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Error Absoluto (miles USD)', rotation=270, labelpad=20)
        
        # Estadísticas del modelo
        r2 = self.results[self.best_model_name]['R2']
        mae = self.results[self.best_model_name]['MAE']
        rmse = self.results[self.best_model_name]['RMSE']
        
        stats_text = f'Modelo: {self.best_model_name}\n'
        stats_text += f'R²: {r2:.4f}\n'
        stats_text += f'MAE: ${mae:.2f}k\n'
        stats_text += f'RMSE: ${rmse:.2f}k\n'
        stats_text += f'Muestras: {len(self.y_test)}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Análisis de residuos
        residuals = self.y_test - self.y_pred
        residual_stats = f"Análisis de Residuos: Media: {residuals.mean():.3f}, "
        residual_stats += f"Desv. Std: {residuals.std():.3f}, "
        residual_stats += f"Max Error: ${abs(residuals).max():.2f}k"
        
        residual_label = ttk.Label(self.plot_frame, text=residual_stats,
                                  font=("Arial", 10), foreground="darkblue")
        residual_label.pack(pady=5)
    
    def run(self):
        """Ejecutar la aplicación"""
        self.root.mainloop()

if __name__ == '__main__':
    app = BostonHousingRegressionDashboard()
    app.run()