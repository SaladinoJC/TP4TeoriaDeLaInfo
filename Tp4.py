import sys
import numpy as np
from math import log2

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

def cargar_datos_archivo(filename):
    """Carga el contenido de un archivo binario."""
    try:
        with open(filename, "rb") as f:
            data = f.read()
        return data
    except FileNotFoundError:
        print(f"Error: El archivo {filename} no se encontró.")
        sys.exit(1)

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
            print("errores de fila")
            print(errores_fila)
            print("errores de columna")
            print(errores_columna)
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

def calcular_entropia_posteriori(matriz_probabilidades, probabilidades):
    H_cond = 0
    for i in range(2):
        for j in range(2):
            if matriz_probabilidades[i, j] > 0:
                H_cond -= probabilidades[i] * matriz_probabilidades[i, j] * log2(matriz_probabilidades[i, j])
    return H_cond

def calcular_informacion_mutua(H_apriori, H_posteriori):
    return H_apriori - H_posteriori

def calcular_perdida(matriz_probabilidades, probabilidades):
    H_cond = 0
    prob_y = [sum(matriz_probabilidades[:, j] * probabilidades) for j in range(2)]

    for j in range(2):
        for i in range(2):
            if prob_y[j] > 0 and matriz_probabilidades[i, j] > 0:
                H_cond -= prob_y[j] * (matriz_probabilidades[i, j] / prob_y[j]) * log2(matriz_probabilidades[i, j] / prob_y[j])
    return H_cond


def main():
    if len(sys.argv) != 4:
        print("Uso: python Tpi4.py sent received N")
        sys.exit(1)
    
    sent_file = sys.argv[1]
    received_file = sys.argv[2]
    N = int(sys.argv[3])

    sent_data = cargar_datos_archivo(sent_file)
    received_data = cargar_datos_archivo(received_file)

    probabilidades = calcular_probabilidades(sent_data)

    entropia_sent = calcular_entropia(probabilidades)
    print(f"Entropía del archivo 'sent': {entropia_sent:.4f} bits")

    matrices_con_paridad = matriz_paridad_cruzada(sent_data, N)
    print("Matrices con paridad cruzada generadas a partir de 'sent':")
    for idx, matriz in enumerate(matrices_con_paridad):
        print(f"Matriz {idx + 1}:\n{matriz}")

    num_bloques = len(matrices_con_paridad)
    matrices_recibidas = leer_matrices_recibidas(received_data, N, num_bloques)
    print("Matrices leídas de 'received':")
    for idx, matriz in enumerate(matrices_recibidas):
        print(f"Matriz {idx + 1}:\n{matriz}")

    matriz_probabilidades = estimar_matriz_probabilidades(matrices_con_paridad, matrices_recibidas)
    print("Matriz de probabilidades del canal binario:")
    print(matriz_probabilidades)

    resultado= verificar_paridad_cruzada(matrices_recibidas,N)
    print(resultado)

    print("probabilidades")
    print(probabilidades)

    entropia_posteriori = calcular_entropia_posteriori(matriz_probabilidades, probabilidades)
    print(f"Entropía a posteriori (H(Y|X)): {entropia_posteriori:.4f} bits")

    informacion_mutua = calcular_informacion_mutua(entropia_sent, entropia_posteriori)
    print(f"Información mutua (I(X;Y)): {informacion_mutua:.4f} bits")

    perdida = calcular_perdida(matriz_probabilidades, probabilidades)
    print(f"Pérdida (H(X|Y)): {perdida:.4f} bits")
    

if __name__ == "__main__":
    main()
