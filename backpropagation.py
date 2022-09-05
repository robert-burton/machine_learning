import math
import random
from functools import partial

"""
Modulo para entrenamiento de red neuronal utilizando backpropagation
"""
def sigmoide(x):
#Fn de activacion de neurona  
  return 1/(1+math.exp(-x))

def producto_punto(v,w):
#Fn para realizar prod punto entre vector entrada y vector pesos
  return sum(x*y for x,y in zip(v,w))

def salida_neurona(pesos,entrada):
#Ejecuta fn de activacion con pesos y entrada de la neurona
#NOTA: el sesgo ya no se suma, se incluye en prod punto (elemento final de vector de pesos)
  return sigmoide(producto_punto(pesos,entrada))

def random_nn():
#Genera pesos aleatorios entre [-1,1] para la red
#NOTA: elemento final de c/neurona corresponde a sesgo
  random.seed(7)
  rand = partial(random.randint)
  xor_nn = [#capa oculta, 2 neuronas
            [[rand(-100,100)/100, rand(-100,100)/100, rand(-100,100)/100], [rand(-100,100)/100, rand(-100,100)/100, rand(-100,100)/100]],
            #capa salida, 1 neurona
            [[rand(-100,100)/100, rand(-100,100)/100, rand(-100,100)/100]]]
  return xor_nn

def ffnn(red_neuronal, entrada):
#Para c/capa en la red, genera un vector con la salida de cada neurona
  salidas =[]
  for capa in red_neuronal:
    entrada = entrada + [1]
    salida = [salida_neurona(neurona, entrada) for neurona in capa]
    salidas.append(salida)
    entrada = salida
  return salidas

def backpropagation(xor_nn, v_entrada, v_objetivo):
#Obtiene gradientes para modificar pesos en direccion desada

  #Se obtienen salidas de capa oculta y capa salida
  salidas_ocultas, salidas = ffnn(xor_nn, v_entrada)
  
  #Listas para valores modificados de pesos en capas oculta y salida
  salida_nuevo = []
  oculta_nuevo = []
  
  #Tasa de aprendizaje: usualmente buen valor
  alfa = 0.1
  
  #Calculo de error (suma de cuadrados de residuos)
  error = 0.5*sum((salida-objetivo)*(salida-objetivo) for salida, objetivo in zip(salidas, v_objetivo))
  
  #Vector gradiente de capa salida: contiene como elementos las derivadas del error total respecto a cada peso de esta capa
  salida_deltas = [salida*(1-salida)*(salida-objetivo) for salida,objetivo in zip(salidas, v_objetivo)]
  
  #Ciclo para modificar pesos de c/neurona en capa salida
  #Iteramos sobre la capa salida, i nos indica en que neurona de la capa nos encontramos
  for i, neurona_salida in enumerate(xor_nn[-1]):
    #Para cada neurona de salida, se itera sobre vector salida de capa oculta
    #NOTA: se agrega coef. de sesgo al vector salida de capa oculta
    for j, salida_oculta in enumerate(salidas_ocultas+[1]):
      #Se modifican ligeramente el peso j de la neurona i, en direccion opuesta al gradiente
      #Se resta la derivada del error total respecto a c/peso de la neurona en capa salida
      #NOTA: la expresion 'salida_deltas[i]*salida_oculta' corresponde a d_E/dw_j
      #Cada peso j esta conectado a una sola neurona en la capa oculta, por lo que su derivada debe reflejar esto
      #'salida_oculta' corresponde al valor de activacion que arroja la neurona j de la capa oculta
      #con la que el peso j se multiplica para generar el prod punto que sera la entrada de la fn de activacion de la neurona de salida i
      neurona_salida[j] -= salida_deltas[i]*salida_oculta*alfa 
    #Se genera nuevo vector para capa de salida, con pesos/sesgo modificados
    salida_nuevo.append(neurona_salida)

  #Vector gradiente de capa oculta: contiene como elementos las derivadas del error total respecto a cada peso de esta capa
  oculta_deltas = [salida_oculta*(1-salida_oculta)*
                    producto_punto(salida_deltas,[n[i] for n in xor_nn[-1]])
                    for i, salida_oculta in enumerate(salidas_ocultas)]
  
  #Ciclo para modificar pesos de c/neurona en capa oculta
  #Iteramos sobre capa oculta, i nos indica la neurona de la capa en la que estamos
  for i, neurona_oculta in enumerate(xor_nn[0]):
    #Para c/neurona se itera sobre el vector de entrada
    #NOTA: se agrega coef. de sesgo al vector de entrada
    for j, input in enumerate(v_entrada+[1]):
      #Se modifica ligeramente el peso j de la neurona i, en direccion opuesta al gradiente
      #Se resta la derivada del error total respecto a c/peso de la capa oculta
      neurona_oculta[j] -= oculta_deltas[i]*input*alfa
    #Se genera nuevo vector para capa oculta, con pesos/sesgo modificados
    oculta_nuevo.append(neurona_oculta)
  
  return oculta_nuevo, salida_nuevo, error

"""
Programa principal
"""

#Inicializacion de red neuronal con pesos aleatorios para primera iteracion de ciclo backpropagation
xor_nn = random_nn()

#Inicializacion de error e iterador para ciclo de backpropagation
promedio_errores_cuadrados = 1
i = 1

#Ciclo de backpropagation
while promedio_errores_cuadrados > 0.0005:
  
  print("iteracion: ", i)
  
  #Entrenamos a la red para que reconozga que 1 xOR 1 = 0
  #Se normalizan datos entre -1 y 1
  #En este intervalo se encuetra la region sensible de la fn de activacion usada (sigmoide)
  oculta, salida, error1 = backpropagation(xor_nn, [x*2-1 for x in [1,1]], [0])
  xor_nn = [oculta, salida]
  
  oculta, salida, error2 = backpropagation(xor_nn, [x*2-1 for x in [0,0]], [0])
  xor_nn = [oculta, salida]
  
  oculta, salida, error3 = backpropagation(xor_nn, [x*2-1 for x in [1,0]], [1])
  xor_nn = [oculta, salida]
  
  oculta, salida, error4 = backpropagation(xor_nn, [x*2-1 for x in [0,1]], [1])
  xor_nn = [oculta, salida]
  
  promedio_errores_cuadrados = (error1+error2+error3+error4)/4
  print("error: ", promedio_errores_cuadrados)
  
  i = i+1

#Pesos y sesgos de la red entrenada
print(xor_nn)

#Prueba de red XOR entreanada
#Se normaliza la entrada para que tenga valores entre -1 y 1
#En este intervalo se encuetra la region sensible de la fn de activacion usada (sigmoide)
print(ffnn(xor_nn,[x*2-1 for x in [0,0]])[-1])
print(ffnn(xor_nn,[x*2-1 for x in [0,1]])[-1])
print(ffnn(xor_nn,[x*2-1 for x in [1,0]])[-1])
print(ffnn(xor_nn,[x*2-1 for x in [1,1]])[-1])