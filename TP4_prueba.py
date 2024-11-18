import sys
import numpy as np
from math import log2

def cargar_datos_archivo(filename):
    """Carga el contenido de un archivo binario."""
    try:
        with open(filename, "rb") as f:
            data = f.read()
        return data
    except FileNotFoundError:
        print(f"Error: El archivo {filename} no se encontró.")
        sys.exit(1)

def calcular_probabilidades(data):
    total_bits = len(data) * 8
    bit_count = sum(bin(byte).count('1') for byte in data)
    prob_1 = bit_count / total_bits if total_bits > 0 else 0
    prob_0 = 1 - prob_1

    probabilidades = [prob_0, prob_1]
    return probabilidades

def calcular_entropia(prob):
    """Calcula la entropía de una fuente de memoria nula en bits."""
    
    if prob[1] > 0 and prob[0] > 0:
        entropia = -prob[1] * log2(prob[1]) - prob[0] * log2(prob[0])
    else:
        entropia = 0
    return entropia

def matriz_paridad_cruzada(data, N):
    """Divide los datos en bloques de N x N bits y aplica paridad cruzada a cada bloque."""
    bit_array = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    bloques = (len(bit_array) + N**2 - 1) // (N**2)
    bit_array = np.pad(bit_array, (0, bloques * N**2 - len(bit_array)), 'constant')

    matrices_con_paridad = []
    
    for i in range(bloques):
        bloque = bit_array[i * N**2 : (i + 1) * N**2].reshape(N, N)
        filas_paridad = np.append(bloque, np.mod(bloque.sum(axis=1), 2).reshape(N, 1), axis=1)
        columnas_paridad = np.append(filas_paridad, np.mod(filas_paridad.sum(axis=0), 2).reshape(1, N + 1), axis=0)
        matrices_con_paridad.append(columnas_paridad)
    
    return matrices_con_paridad

def leer_matrices_recibidas(data, N, num_bloques):
    """Lee num_bloques de (N+1) x (N+1) bits desde los datos recibidos."""
    total_bits = num_bloques * (N + 1) ** 2
    bit_array = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    if len(bit_array) < total_bits:
        print(f"Advertencia: El archivo 'received' no tiene suficientes bits. Se completará con ceros.")
    bit_array = np.pad(bit_array, (0, max(0, total_bits - len(bit_array))), 'constant')

    matrices_recibidas = []
    for i in range(num_bloques):
        bloque_bits = bit_array[i * (N + 1) ** 2 : (i + 1) * (N + 1) ** 2]
        matriz_recibida = bloque_bits.reshape(N + 1, N + 1)
        matrices_recibidas.append(matriz_recibida)
    
    return matrices_recibidas

def estimar_matriz_probabilidades(matrices_sent, matrices_received):
    """Estima la matriz de probabilidades del canal binario a partir de listas de matrices."""
    count_00 = count_01 = count_10 = count_11 = 0
    
    for matriz_sent, matriz_received in zip(matrices_sent, matrices_received):
        for bit_sent, bit_received in zip(matriz_sent.ravel(), matriz_received.ravel()):
            if bit_sent == 0 and bit_received == 0:
                count_00 += 1
            elif bit_sent == 0 and bit_received == 1:
                count_01 += 1
            elif bit_sent == 1 and bit_received == 0:
                count_10 += 1
            elif bit_sent == 1 and bit_received == 1:
                count_11 += 1

    count0  = count_00 + count_01
    count1  = count_10 + count_11
    prob_00 = count_00 / count0 if count0 > 0 else 0
    prob_01 = count_01 / count0 if count0 > 0 else 0
    prob_10 = count_10 / count1 if count1 > 0 else 0
    prob_11 = count_11 / count1 if count1 > 0 else 0

    return np.array([[prob_00, prob_01], [prob_10, prob_11]]) 

def verificar_paridad_cruzada(matrices, N):
    correctas = 0
    incorrectas = 0
    corregibles = 0

    for matriz in matrices:
        # Verifica filas y columnas de paridad
        filas_ok = np.all(np.mod(matriz[:-1, :-1].sum(axis=1), 2) == matriz[:-1, -1])
        columnas_ok = np.all(np.mod(matriz[:-1, :-1].sum(axis=0), 2) == matriz[-1, :-1])
        paridad_global_ok = matriz[:-1, -1].sum() % 2 == matriz[-1, -1]

        if filas_ok and columnas_ok and paridad_global_ok:
            correctas += 1
        else:
            # Verificar si es corregible
            errores_fila = np.where(np.mod(matriz[:-1, :-1].sum(axis=1), 2) != matriz[:-1, -1])[0]
            errores_columna = np.where(np.mod(matriz[:-1, :-1].sum(axis=0), 2) != matriz[-1, :-1])[0]

            if len(errores_fila) == 1 and len(errores_columna) == 1:
                # Un único bit incorrecto es corregible
                corregibles += 1
            else:
                incorrectas += 1

    return {
        "correctas": correctas,
        "incorrectas": incorrectas,
        "corregibles": corregibles
    }

def calcular_priabilidades_bj(p_bi_ai, p_ai):
    n = len(p_ai)
    m = len(p_bi_ai[0])

    p_bj = [0] * m

    for j in range(m):
        for i in range(n):
            p_bj[j] += p_ai[i] * p_bi_ai[i][j]
    
    return p_bj

def calcular_probabilidades_ai_bj(p_bj_ai, p_ai, p_bj):
    """
    Calcula la matriz de probabilidades P(ai|bj) a partir de P(bj|ai), P(ai) y P(bj).

    :param p_bj_ai: Matriz de probabilidades condicionales P(bj|ai) (lista de listas).
    :param p_ai: Vector de probabilidades a priori P(ai).
    :param p_bj: Vector de probabilidades P(bj).
    :return: Matriz de probabilidades a posteriori P(ai|bj).
    """
    n = len(p_ai)  # Número de eventos posibles de ai (r)
    m = len(p_bj)  # Número de eventos posibles de bj (s)

    # Inicializamos la matriz P(ai|bj) con ceros
    p_ai_bj = [[0 for _ in range(m)] for _ in range(n)]

    # Calculamos P(ai|bj) para cada par (i, j)
    for j in range(m):
        for i in range(n):
            p_ai_bj[i][j] = (p_bj_ai[i][j] * p_ai[i]) / p_bj[j]

    return p_ai_bj

def calcular_entropia_posteriori(p_ai_bj):
    """
    Calcula la entropía a posteriori H(A|bj) para cada bj dado P(ai|bj).

    :param p_ai_bj: Matriz de probabilidades condicionales P(ai|bj) (lista de listas).
    :return: Lista con los valores de entropía H(A|bj) para cada bj.
    """
    m = len(p_ai_bj[0])  # Número de columnas (eventos bj)
    entropias = []

    for j in range(m):
        entropia = 0
        for i in range(len(p_ai_bj)):
            p_aibj = p_ai_bj[i][j]
            if p_aibj > 0:  # Evitamos log(0)
                entropia += p_aibj * log2(1 / p_aibj)
        entropias.append(entropia)
    
    return entropias

def multiplicar_matriz_vector(matriz, vector):
    """
    Multiplica una matriz por un vector.

    :param matriz: Lista de listas representando la matriz (m x n).
    :param vector: Lista representando el vector (n).
    :return: Lista representando el resultado de la multiplicación (m).
    """
    if len(matriz[0]) != len(vector):
        raise ValueError("El número de columnas de la matriz debe coincidir con el tamaño del vector.")
    
    resultado = []
    for fila in matriz:
        # Producto escalar entre la fila de la matriz y el vector
        producto = sum(f * v for f, v in zip(fila, vector))
        resultado.append(producto)
    
    return resultado

def calcular_probabilidad_conjunta(P_condicional, P_bj):
    """
    Calcula la matriz de probabilidad conjunta P(A_i, B_j).

    Parámetros:
    - P_condicional: Matriz de probabilidades condicionales P(A_i | B_j), de tamaño m x n.
    - P_bj: Vector de probabilidades P(B_j), de tamaño n.

    Retorna:
    - Matriz conjunta P(A_i, B_j) de tamaño m x n.
    """
    # Convertir los parámetros a arrays de numpy para facilitar las operaciones
    P_condicional = np.array(P_condicional)
    P_bj = np.array(P_bj)

    # Validar que el número de columnas de P_condicional coincida con el tamaño de P_bj
    if P_condicional.shape[1] != len(P_bj):
        raise ValueError("El tamaño del vector P(B_j) no coincide con el número de columnas de P_condicional")

    # Calcular la matriz conjunta
    P_conjunta = P_condicional * P_bj

    return P_conjunta


def calcular_entropia_posteriori_media(P_conjunta, P_condicional):
    """
    Calcula la entropía media a posteriori H(A|B).

    Parámetros:
    - P_conjunta: Matriz de probabilidades conjuntas P(A, B), de tamaño m x n.
    - P_condicional: Matriz de probabilidades condicionales P(A | B), de tamaño m x n.

    Retorna:
    - La entropía media a posteriori H(A | B).
    """
    # Convertir los parámetros a arrays de numpy para facilitar las operaciones
    P_conjunta = np.array(P_conjunta)
    P_condicional = np.array(P_condicional)

    # Calcular la probabilidad marginal P(B_j) sumando las filas de la matriz conjunta
    P_bj = np.sum(P_conjunta, axis=0)

    # Inicializar la entropía
    entropia_posteriori = 0

    # Iterar sobre todas las columnas (B_j) y filas (A_i) para calcular H(A|B)
    for j in range(P_condicional.shape[1]):
        for i in range(P_condicional.shape[0]):
            if P_condicional[i, j] > 0:  # Evitar logaritmos de 0
                entropia_posteriori += P_bj[j] * P_condicional[i, j] * np.log2(1 / P_condicional[i, j])

    return entropia_posteriori

def calcular_informacion_mutua(P_conjunta, P_a, P_b):
    """
    Calcula la información mutua I(A, B).

    Parámetros:
    - P_conjunta: Matriz de probabilidades conjuntas P(A, B), de tamaño m x n.
    - P_a: Vector de probabilidades marginales P(A), de tamaño m.
    - P_b: Vector de probabilidades marginales P(B), de tamaño n.

    Retorna:
    - La información mutua I(A, B).
    """
    # Convertir los parámetros a arrays de numpy para facilitar las operaciones
    P_conjunta = np.array(P_conjunta)
    P_a = np.array(P_a)
    P_b = np.array(P_b)

    # Inicializar la información mutua
    informacion_mutua = 0

    # Iterar sobre todas las filas (A_i) y columnas (B_j)
    for i in range(P_conjunta.shape[0]):
        for j in range(P_conjunta.shape[1]):
            if P_conjunta[i, j] > 0:  # Evitar logaritmos de 0
                informacion_mutua += P_conjunta[i, j] * np.log2(P_conjunta[i, j] / (P_a[i] * P_b[j]))

    return informacion_mutua

def imprimir_matrices(matrices):
    for idx, matriz in enumerate(matrices):
        print(f"Matriz {idx + 1}:\n{matriz}")

    return 

def calcular_perdida(P_conjunta, P_condicional_bj_ai):
    """
    Calcula la pérdida del canal H(B|A).

    Parámetros:
    - P_conjunta: Matriz de probabilidades conjuntas P(A, B), de tamaño m x n.
    - P_condicional_bj_ai: Matriz de probabilidades condicionales P(B_j | A_i), de tamaño m x n.

    Retorna:
    - La pérdida del canal H(B | A).
    """
    # Convertir los parámetros a arrays de numpy para facilitar las operaciones
    P_conjunta = np.array(P_conjunta)
    P_condicional_bj_ai = np.array(P_condicional_bj_ai)

    # Calcular la probabilidad marginal P(A_i) sumando las columnas de la matriz conjunta
    P_ai = np.sum(P_conjunta, axis=1)

    # Inicializar la entropía a posteriori
    perdida = 0

    # Iterar sobre todas las filas (A_i) y columnas (B_j)
    for i in range(P_condicional_bj_ai.shape[0]):
        for j in range(P_condicional_bj_ai.shape[1]):
            if P_condicional_bj_ai[i, j] > 0:  # Evitar logaritmos de 0
                perdida += P_ai[i] * P_condicional_bj_ai[i, j] * np.log2(1 / P_condicional_bj_ai[i, j])

    return perdida

def calcular_entropias_condicionales(P_condicional_bj_ai):
    """
    Calcula las entropías condicionales H(B|A_i) para cada A_i.

    Parámetros:
    - P_condicional_bj_ai: Matriz de probabilidades condicionales P(B_j | A_i), de tamaño m x n.

    Retorna:
    - Lista de entropías condicionales H(B | A_i) para cada A_i.
    """
    # Convertir la matriz a un array de numpy
    P_condicional_bj_ai = np.array(P_condicional_bj_ai)

    # Inicializar la lista para almacenar las entropías condicionales
    entropias_condicionales = []

    # Iterar sobre cada fila (A_i) de la matriz condicional
    for i in range(P_condicional_bj_ai.shape[0]):
        entropia_ai = 0
        for j in range(P_condicional_bj_ai.shape[1]):
            if P_condicional_bj_ai[i, j] > 0:  # Evitar logaritmos de 0
                entropia_ai += P_condicional_bj_ai[i, j] * np.log2(1 / P_condicional_bj_ai[i, j])
        
        # Agregar la entropía condicional de A_i a la lista
        entropias_condicionales.append(entropia_ai)

    return entropias_condicionales

def main():
    if len(sys.argv) != 4:
        print("Uso: python Tpi4.py sent received N")
        sys.exit(1)
    
    sent_file = sys.argv[1]
    received_file = sys.argv[2]
    N = int(sys.argv[3])

    sent_data = cargar_datos_archivo(sent_file)
    received_data = cargar_datos_archivo(received_file)

    p_ai = calcular_probabilidades(sent_data)
    print("Las probabilidades P(ai) son: ")
    print(p_ai)
    print()

    #------------------------------------------------------------Inciso a
    print("La entropia de la fuente es H(S): ")
    print(calcular_entropia(p_ai))
    print()

    #------------------------------------------------------------Inciso b
    matrices_paridad = matriz_paridad_cruzada(sent_data, N)
    #print("Matrices con paridad cruzada generadas a partir de 'sent':")
    #imprimir_matrices(matrices_paridad)


    num_bloques = len(matrices_paridad)
    matrices_recibidas = leer_matrices_recibidas(received_data, N, num_bloques)
    #print("Matrices leídas de 'received':")
    #imprimir_matrices(matrices_recibidas)


    #------------------------------------------------------------Inciso c
    p_bj_ai = estimar_matriz_probabilidades(matrices_paridad, matrices_recibidas)

    print("Matriz de probabilidad P(bj/ai): ")
    print(p_bj_ai)
    print()
    #------------------------------------------------------------Inciso d
    resultado= verificar_paridad_cruzada(matrices_recibidas,N)
    print(resultado)
    print()

    p_bj = calcular_priabilidades_bj(p_bj_ai, p_ai)
    print("Las probabilidades P(bj) son: ")
    print([float(x) for x in p_bj])
    print()

    p_ai_bj = calcular_probabilidades_ai_bj(p_bj_ai, p_ai, p_bj)
    print("Matriz de probabilidad P(ai/bj): ")
    for fila in p_ai_bj:
        print([float(x) for x in fila])
    print()

    entropias_posteriori = calcular_entropia_posteriori(p_ai_bj)
    print("Entropias a posteriori H(A/bj):")
    print([round(float(x), 4) for x in entropias_posteriori])
    print()

    h_b_ai = calcular_entropias_condicionales(p_bj_ai)
    print("Entropias a posteriori H(B/ai):")
    print([round(float(x), 4) for x in h_b_ai])
    print()

    probabilidad_simultanea1 = calcular_probabilidad_conjunta(p_ai_bj,p_bj)
    probabilidad_simultanea2 = calcular_probabilidad_conjunta(p_bj_ai,p_ai)
    
    print("Probabilidades simultaneas P(ai,bj)")
    print(probabilidad_simultanea1)
    print()

    #print("Probabilidades simultaneas 2")
    #print(probabilidad_simultanea2)

    ruido = calcular_entropia_posteriori_media(probabilidad_simultanea1, p_ai_bj)
    print(f"Entropía media a posteriori H(A|B) o equivocacion/ruido: {ruido:.4f} bits")
    print()

    informacion_mutua = calcular_informacion_mutua(probabilidad_simultanea1, p_ai, p_bj)
    print(f"Información mutua I(A, B): {informacion_mutua:.4f} bits")
    print()

    print("La entropia H(B): ")
    print(calcular_entropia(p_bj))
    print()

    perdida = calcular_perdida(probabilidad_simultanea1, p_bj_ai)
    print(f"Entropía media a posteriori H(B|A) o perdida: {perdida:.4f} bits")
    print()

if __name__ == "__main__":
    main()