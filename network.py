import random
import math

class Red_Neuronal:
    """
    Red neuronal prealimentada (FFNN) con una capa oculta.
    La red cuenta con metodos para entrenamiento y predicciones.

    La red tiene como meta ser entrenada para reconocimiento de imagenes del dataset MNIST.
    Las entrada esperada del modelo es una lista de valores enteros, por ejemplo [0, 1, 255...].
    Entradas cuyos elementos sean listas, por ejemplo [[255,0,0], [100,100,100]...] quedan fuera del alcance de este modelo.

    Como en el dataset MNIST la etiqueta de c/entrada corresponde al elemento 0, si se quiere usar con otro dataset debe
    asegurarse que c/entrada siga el mismo formato.
    """
    def __init__(self, tamano_entradas: int, neuronas_capa_oculta: int, neuronas_capa_salida: int, tasa_aprendizaje=0.1):
        #Las capas estan compuestas de 'neuronas' (listas de pesos + sesgo)
        #Se almacena el error actual del modelo y la ultima salida arrojada
        self.entradas = tamano_entradas
        self.capa_oculta = [[random.randint(-100,100)/100 for i in range(self.entradas+1)] for j in range(neuronas_capa_oculta)]
        self.capa_salida = [[random.randint(-100,100)/100 for i in range(neuronas_capa_oculta+1)] for j in range(neuronas_capa_salida)]
        self.alfa = tasa_aprendizaje
        self.objetivos = [] #lista que contiene vectores objetivos (listas de enteros)
    
    def sigmoide(self, x: float):
        #Fn de activacion para neuronas
        return 1/(1+math.exp(-x))

    def producto_punto(self, v: list, w: list):
        #Fn para realizar prod punto entre vector entrada y vector pesos
        return sum(x*y for x,y in zip(v,w))
    
    def definir_objetivo(self, objetivos: list):
        #NOTA : EL OBJETIVO DEBERIA SER UNA LISTA DE LISTAS CON LOS VALORES DE ACTIVACION DE LA CAPA SALIDA
        #Cada lista elemento del argumento objetivos debe ser del mismo tamaño que la capa de salida
        #[[0,0,0],
        #[0,0,1], ...]
        for i in range(len(self.capa_salida)):
            if len(objetivos[i]) != len(self.capa_salida):
                print(f"Tamaño de vector objetivo no corresponde al numero de salidas de la red ({len(self.capa_salida)})")
                return
        self.objetivos = objetivos

    def calcular_error(self, salida: list, objetivo: list):
        #Calcula error entre la salida producida por cierta entrada de un set y su vector objetivo correspondiente
        #Para uso iterativo en alimentar_red()
        #return sum(math.pow(a-y,2) for a, y in zip(salida, objetivo))/2
        return -1 * sum(y * math.log(a) + (1 - y) * math.log(1 - a) for a, y in zip(salida, objetivo))
    
    def modificar_capa_oculta(self, operacion: str, numero_neuronas: int):
        #Agregar o eliminar neuronas a la capa oculta
        match operacion:
            case '+':
                for i in range(numero_neuronas):
                    self.capa_oculta.append([random.randint(-100,100)/100 for i in range(self.entradas+1)])
                    for j in range(len(self.capa_salida)):
                        self.capa_salida[j].append(random.randint(-100,100)/100)
            case '-':
                for i in range(numero_neuronas):
                    self.capa_oculta.pop(-1)
                    for j in range (len(self.capa_salida)):
                        self.capa_salida[j].pop(-1)
            case _:
                print("Error: especificar operacion valida como primer argumento ('+', '-')...")
    
    def reinicializar(self):
        #Reinicializa los pesos sinapticos con valores aleatorios
        for i in range(len(self.capa_oculta)):
            for j in range(len(self.capa_oculta[i])):
                self.capa_oculta[i][j] = random.randint(-100,100)/100
        for i in range(len(self.capa_salida)):
            for j in range(len(self.capa_salida[i])):
                self.capa_salida[i][j] = random.randint(-100,100)/100
        self.error = 1
    
    def alimentar_red(self, entrada: list, objetivo: list):
        #Recibe un vector entrada del set y regresa una lista con el vector de activaciones de cada capa
        #Para uso en entrenar_red() y prediccion()
        #Se evaluda primero que el modelo cuente con vector objetivo, para calcular error de cada neurona de salida
        if self.objetivos != []:
            #Realiza algoritmo feedforward, activando las neuronas y regresando un vector con las salidas del modelo en su estado actual
            activacion_capa_oculta = [self.sigmoide(self.producto_punto(entrada+[1], neurona_oculta)) for neurona_oculta in self.capa_oculta] #salida capa oculta
            activacion_capa_salida = [self.sigmoide(self.producto_punto(activacion_capa_oculta+[1], neurona_salida)) for neurona_salida in self.capa_salida] #salida final
            error = self.calcular_error(activacion_capa_salida, objetivo)
            return [activacion_capa_oculta, activacion_capa_salida, error]
        else:
            print("No se ha definido un vector objetivo. Usar metodo .definir_objetivo() antes de alimentar...")

    def backpropagation(self, entrada: list, objetivo: list, activacion_capa_oculta: list, activacion_capa_salida: list):
        #Obtiene vector de derivadas parciales del error total respecto a c/peso
        #Actualiza las capas oculta y de salida utilizando vector de derivadas
        nueva_capa_salida = self.capa_salida
        nueva_capa_oculta = self.capa_oculta

        #Descenso de gradiente en capa salida
        delta_activacion_capa_salida = []
        activacion_capa_oculta.append(1)
        for i, neurona_salida in enumerate(self.capa_salida):
            delta_activacion_capa_salida.append(-1 * (objetivo[i]-activacion_capa_salida[i]) * activacion_capa_salida[i] * (1-activacion_capa_salida[i]))
            for j, peso in enumerate(neurona_salida): #iteramos sobre los pesos de c/neurona                
                try:
                    delta_peso = delta_activacion_capa_salida[i] * activacion_capa_oculta[j]
                    #nueva_capa_salida[i][j] = nueva_capa_salida[i][j] - (self.alfa * delta_peso)
                    nueva_capa_salida[i][j] = peso - (self.alfa * delta_peso)
                except:
                    print(len(activacion_capa_oculta))

        #Descenso de gradiente en capa oculta
        delta_activacion_capa_oculta = []
        entrada.append(1)
        for i, neurona_oculta in enumerate(self.capa_oculta):
            #delta_activacion_capa_oculta.append(-1 * activacion_capa_oculta[i] * (1-activacion_capa_oculta[i]) * self.producto_punto(delta_activacion_capa_salida, [x[i] for x in self.capa_salida]))
            delta_activacion_capa_oculta.append(-1 * activacion_capa_oculta[i] * (1-activacion_capa_oculta[i]) * self.producto_punto(delta_activacion_capa_salida, [w[i] for w in self.capa_oculta]))
            for j, peso in enumerate(neurona_oculta):                
                delta_peso = delta_activacion_capa_oculta[i] * entrada[j]
                nueva_capa_oculta[i][j] = peso - (self.alfa * delta_peso)
        
        #Actualizacion de pesos en modelo
        self.capa_oculta = nueva_capa_oculta
        self.capa_salida = nueva_capa_salida
    
    def entrenar_red(self, set_entrenamiento: list, batch=50):
        #Implementar ciclos de backpropagation para minimizar fn de error
        #Requiere como argumento un 'set_entrenamiento' de n elementos, cuyos elementos sean del mismo tamaño que la entrada del modelo
        error_promedio = 1
        error_promedio_meta = 0.1 #error original: 0.0005
        errores = []
        iteracion = 1
        while error_promedio > error_promedio_meta and iteracion <= 100000: 
            subset_entrenamiento = random.sample(set_entrenamiento, batch)
            for entrada in subset_entrenamiento:
                digito_objetivo = entrada[0]
                activacion_capa_oculta, activacion_capa_salida, error_i = self.alimentar_red(entrada[1:], self.objetivos[digito_objetivo])
                errores.append(error_i)
                self.backpropagation(entrada[1:], self.objetivos[digito_objetivo], activacion_capa_oculta, activacion_capa_salida)
            error_promedio = sum(errores)/len(errores)
            print(f"Iteracion: {iteracion},\tPromedio de errores cuadrados: {error_promedio}")
            iteracion = iteracion + 1
        if error_promedio <= error_promedio_meta: 
            print(f"**Entrenamiento terminado**\nIteraciones: {iteracion}\nPromedio de errores cuadrados: {error_promedio}\n")
        else:
            print(f"**Modelo no converge despues de {iteracion} iteraciones...**")
        return
    
    def prediccion(self, set_validacion: list, batch=0):
        #Entrada: set de validacion con n elementos
        #se muestra digito ingresado, digito detectado y confianza
        aciertos = 0
        precision_modelo = 0
        if batch <= 0:
            for entrada in set_validacion:
                digito_objetivo = entrada[0]
                _o, salida, _e = self.alimentar_red(entrada[1:], self.objetivos[digito_objetivo])
                digito_detectado = salida.index(max(salida))
                confianza = max(salida)
                print(f"Ingresado: {digito_objetivo}\tDetectado: {digito_detectado}\tConfianza: {confianza}")
                if digito_objetivo == digito_detectado:
                    aciertos += 1
        else:
            set_validacion = random.sample(set_validacion, batch)
            for entrada in set_validacion:
                digito_objetivo = entrada[0]
                _o, salida, _e = self.alimentar_red(entrada[1:], self.objetivos[digito_objetivo])
                digito_detectado = salida.index(max(salida))
                confianza = max(salida)
                print(f"Ingresado: {digito_objetivo}\tDetectado: {digito_detectado}\tConfianza: {confianza}")
                if digito_objetivo == digito_detectado:
                    aciertos += 1
        precision_modelo = aciertos/len(set_validacion)
        print(f"\nPRECISION DEL MODELO: {precision_modelo}")
        return

"""
Prueba

VECTORES DE ENTRADA MNIST TIENE VALORES ENTRE (0,255)
NORMALIZAR ENTRADAS PARA QUE TENGAN MEDIA(VALOR ESPERADO) 0 Y VARIANZA 1 [-1,1]
"""

#Imporatacion/tratamiento de datos
import os
import pandas as pd
cwd = os.path.dirname(os.path.realpath(__file__))
df_train = pd.read_csv(cwd+'\\mnist_train.csv')
data_train = df_train.values.tolist()
data_train_normalizado = [[int(valor)/255*2-1 if i!=0 else valor for i,valor in enumerate(data_train[j])] for j,entrada in enumerate(data_train)]

#Definicion de red
red_entrada = len(data_train_normalizado[0])-1
red_capa_oculta = 45 #modificable
red_capa_salida = 10 #clasificiones: [0,1,2,...,9]
objetivos_mnist = [[1 if i==j else 0 for i in range(10)] for j in range(10)]
ffnn_mnist = Red_Neuronal(red_entrada,red_capa_oculta,red_capa_salida,0.01)
ffnn_mnist.definir_objetivo(objetivos_mnist)

#Entrenamiento
ffnn_mnist.entrenar_red(data_train_normalizado,256)

#Prediccion
df_test = pd.read_csv(cwd+'\\mnist_test.csv')
data_test = df_test.values.tolist()
data_test_normalizado = [[int(valor)/255*2-1 if i!=0 else valor for i,valor in enumerate(data_test[j])] for j,entrada in enumerate(data_test)]
ffnn_mnist.prediccion(data_train_normalizado, 100)