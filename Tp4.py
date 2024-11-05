import sys
import numpy as np
from math import log2

def calcular_entropia(data):
    """Calcula la entropía de una fuente de memoria nula en bits."""
    total_bits = len(data) * 8
    bit_count = sum(bin(byte).count('1') for byte in data)
    prob_1 = bit_count / total_bits if total_bits > 0 else 0
    prob_0 = 1 - prob_1
    if prob_1 > 0 and prob_0 > 0:
        entropia = -prob_1 * log2(prob_1) - prob_0 * log2(prob_0)
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

def main():
    if len(sys.argv) != 4:
        print("Uso: python Tpi4.py sent received N")
        sys.exit(1)
    
    sent_file = sys.argv[1]
    received_file = sys.argv[2]
    N = int(sys.argv[3])

    sent_data = cargar_datos_archivo(sent_file)
    received_data = cargar_datos_archivo(received_file)

    entropia_sent = calcular_entropia(sent_data)
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

if __name__ == "__main__":
    main()
