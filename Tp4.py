import sys
import numpy as np
from math import log2
from collections import Counter

def calcular_entropia(data):
    """Calcula la entropía de una fuente de memoria nula en bits."""
    total_bits = len(data) * 8
    bit_count = sum(bin(byte).count('1') for byte in data)  # Contar todos los bits '1'
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

def crear_matriz_con_paridad(data, N):
    """Convierte los datos en matrices de tamaño NxN aplicando paridad cruzada."""
    bits = ''.join(format(byte, '08b') for byte in data)
    matrices = []
    for i in range(0, len(bits), N * N):
        matrix_data = bits[i:i + N * N]
        # Completa con ceros si faltan elementos
        matrix_data = matrix_data.ljust(N * N, '0')
        matrix = np.array([int(bit) for bit in matrix_data], dtype=int).reshape(N, N)
        
        # Añadir columna de paridad (paridad de filas)
        parity_column = matrix.sum(axis=1) % 2
        matrix = np.hstack([matrix, parity_column[:, None]])
        
        # Añadir fila de paridad (paridad de columnas)
        parity_row = matrix.sum(axis=0) % 2
        matrix = np.vstack([matrix, parity_row])
        
        matrices.append(matrix)
    return matrices


def guardar_matrices(matrices, filename):
    """Guarda las matrices con paridad en un archivo para visualización o pruebas."""
    with open(filename, "w") as f:
        for idx, matrix in enumerate(matrices):
            f.write(f"Matriz {idx + 1} con paridad:\n")
            for row in matrix:
                f.write(' '.join(map(str, row)) + '\n')
            f.write('\n')

def main():
    # Verificar argumentos de la línea de comandos
    if len(sys.argv) != 4:
        print("Uso: python Tpi4.py sent received N")
        sys.exit(1)
    
    # Leer archivos y parámetro N
    sent_file = sys.argv[1]
    received_file = sys.argv[2]
    N = int(sys.argv[3])

    # Cargar datos de archivos
    sent_data = cargar_datos_archivo(sent_file)
    received_data = cargar_datos_archivo(received_file)

    # Calcular entropía
    entropia_sent = calcular_entropia(sent_data)
    print(f"Entropía del archivo 'sent': {entropia_sent:.4f} bits")

    # Crear matrices con paridad cruzada
    matrices_sent = crear_matriz_con_paridad(sent_data, N)
    matrices_received = crear_matriz_con_paridad(received_data, N)

    # Guardar matrices en archivos para verificar (opcional)
    guardar_matrices(matrices_sent, "matrices_sent.txt")
    guardar_matrices(matrices_received, "matrices_received.txt")

    print("Matrices con paridad guardadas en archivos 'matrices_sent.txt' y 'matrices_received.txt'.")

if __name__ == "__main__":
    main()
