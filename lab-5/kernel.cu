#include <iostream>
#include <cuda_runtime.h>

#define NUM_NEURONAS 10 
#define NUM_ENTRADAS 25      
#define NUM_PATRONES 10 //10 numeros a analizar (0-9)     
#define UMBRAL     0.5f      
#define TASA_APRENDIZAJE 1.0f
#define BIAS_ENTRADA 1.0f

// Para obtener por ejemplo el numero 0, con los 10 numeros(0-9) se debe analizar los pesos(NUM_ENTRADAS) de la primera neurona para su vector sea (1,0,0,0,0,0,0,0,0,0)  
// Para obtener por ejemplo el numero 1, con los 10 numeros(0-9) se debe analizar los pesos(NUM_ENTRADAS) de la segunda neurona para su vector sea (0,1,0,0,0,0,0,0,0,0)  
// Teniendo la matriz de pesos (NUM_NEURONAS*NUM_ENTRADAS)
// En el GPU se debe analizar por threads los vectores de las neuronas. Ya que cada neurona posee una matriz unica de pesos independiente a las demas neuronas. Pudiendo manejarse por thread la actualizacion del vector de pesos de una neurona
//el bias tambien tiene su vector de pesos y deben aplicarse a la formula con su actualizacion de pesos correspondiente.

float entradasCPU[NUM_PATRONES][NUM_ENTRADAS] = {
    // 0
    {1,1,1,1,1,
    1,0,0,0,1,
    1,0,0,0,1,
    1,0,0,0,1,
    1,1,1,1,1},
     // 1
    {0,0,1,0,0,
    0,0,1,0,0,
    0,0,1,0,0,
    0,0,1,0,0,
    0,0,1,0,0},
    // 2
    {1,1,1,1,1,
    0,0,0,0,1,
    1,1,1,1,1,
    1,0,0,0,0,
    1,1,1,1,1},
    // 3
    {1,1,1,1,1,
    0,0,0,0,1,
    0,1,1,1,1,
    0,0,0,0,1,
    1,1,1,1,1},
    // 4
    {1,0,0,0,1,
    1,0,0,0,1,
    1,1,1,1,1,
    0,0,0,0,1,
    0,0,0,0,1},
    // 5
    {1,1,1,1,1,
    1,0,0,0,0,
    1,1,1,1,1,
    0,0,0,0,1,
    1,1,1,1,1},
    // 6
    {1,1,1,1,1,
    1,0,0,0,0,
    1,1,1,1,1,
    1,0,0,0,1,
    1,1,1,1,1},
    // 7
    {1,1,1,1,1,
    0,0,0,0,1,
    0,0,0,0,1,
    0,0,0,0,1,
    0,0,0,0,1},
    // 8
    {1,1,1,1,1,
    1,0,0,0,1,
    1,1,1,1,1,
    1,0,0,0,1,
    1,1,1,1,1},
    // 9
    {1,1,1,1,1,
    1,0,0,0,1,
    1,1,1,1,1,
    0,0,0,0,1,
    1,1,1,1,1}
};

float deseadoCPU[NUM_PATRONES][NUM_NEURONAS] = {
    {1,0,0,0,0,0,0,0,0,0}, // 0
    {0,1,0,0,0,0,0,0,0,0}, // 1
    {0,0,1,0,0,0,0,0,0,0}, // 2
    {0,0,0,1,0,0,0,0,0,0}, // 3
    {0,0,0,0,1,0,0,0,0,0}, // 4
    {0,0,0,0,0,1,0,0,0,0}, // 5
    {0,0,0,0,0,0,1,0,0,0}, // 6
    {0,0,0,0,0,0,0,1,0,0}, // 7
    {0,0,0,0,0,0,0,0,1,0}, // 8
    {0,0,0,0,0,0,0,0,0,1}  // 9
};

// Variables GPU
float* entradasGPU = nullptr; 
float* deseadoGPU = nullptr; 
float* pesosGPU = nullptr; 
float* pesosBiasGPU = nullptr; 
int* errorGPU = nullptr; // un indicador 0 = sin error, 1 = actualizar pesos

// Calcula la salida de las 10 neuronas para un patrón dado y actualiza los pesos
__global__ void entrenamientoGPU(int patronIdx, float* pesosGPU, float* pesosBiasGPU,
    float* entradasGPU, float* deseadoGPU, int* errorGPU) {
    int i = threadIdx.x;  

    // suma de pesos*entredas
    float suma = pesosBiasGPU[i] * BIAS_ENTRADA;
    int offset = patronIdx * NUM_ENTRADAS;  // indice del inicio del patron en el array de entradas
    for (int j = 0; j < NUM_ENTRADAS; ++j) {
        suma += pesosGPU[i * NUM_ENTRADAS + j] * entradasGPU[offset + j];
    }
    float salida = (suma >= UMBRAL) ? 1.0f : 0.0f; //redondeo de la salida

    // Ver si hay error con la igualdad en el deseado y la salida obtenida
    float error = deseadoGPU[patronIdx * NUM_NEURONAS + i] - salida;
    if (error != 0.0f) {
        *errorGPU = 1;
        // actulizar los pesos de la neurona i
        for (int j = 0; j < NUM_ENTRADAS; ++j) {
            // peso_nuevo = peso_viejo + tasa * error * entrada
            pesosGPU[i * NUM_ENTRADAS + j] += TASA_APRENDIZAJE * error * entradasGPU[offset + j];
        }
        // Actualizar el peso del bias de la neurona i
        pesosBiasGPU[i] += TASA_APRENDIZAJE * error * BIAS_ENTRADA;
    }
}

// calcula la salida de las 10 neuronas con los pesos actualizados
__global__ void calcularSumaGPU(int patronIdx, float* pesosGPU, float* pesosBiasGPU,
    float* entradasGPU, float* salidasGPU) {
    int i = threadIdx.x;  
    float suma = pesosBiasGPU[i] * BIAS_ENTRADA;
    int offset = patronIdx * NUM_ENTRADAS;
    for (int j = 0; j < NUM_ENTRADAS; ++j) {
        suma += pesosGPU[i * NUM_ENTRADAS + j] * entradasGPU[offset + j];
    }
    salidasGPU[patronIdx * NUM_NEURONAS + i] = (suma > UMBRAL) ? 1.0f : 0.0f;
}

int main() {
    // ======================================================================================================
    // reservar memoria en GPU 
    cudaMalloc((void**)&entradasGPU, NUM_PATRONES * NUM_ENTRADAS * sizeof(float));
    cudaMalloc((void**)&deseadoGPU, NUM_PATRONES * NUM_NEURONAS * sizeof(float));
    cudaMalloc((void**)&pesosGPU, NUM_NEURONAS * NUM_ENTRADAS * sizeof(float));
    cudaMalloc((void**)&pesosBiasGPU, NUM_NEURONAS * sizeof(float));
    cudaMalloc((void**)&errorGPU, sizeof(int));

    // copiar desde CPU a GPU
    cudaMemcpy(entradasGPU, entradasCPU, sizeof(entradasCPU), cudaMemcpyHostToDevice);
    cudaMemcpy(deseadoGPU, deseadoCPU, sizeof(deseadoCPU), cudaMemcpyHostToDevice);

    // Inicializar pesos y bias
    cudaMemset(pesosGPU, 0, NUM_NEURONAS * NUM_ENTRADAS * sizeof(float));
    cudaMemset(pesosBiasGPU, 0, NUM_NEURONAS * sizeof(float));

    // ======================================================================================================
    // ENTRENAMIENTO
    int epocas = 0;
    bool bucle = false;
    while (!bucle) {
        epocas++;
        bucle = true;  // hasta encontrar un error
        for (int p = 0; p < NUM_PATRONES; ++p) {
            int cero = 0;
            cudaMemcpy(errorGPU, &cero, sizeof(int), cudaMemcpyHostToDevice);
            // usar el kernel para los hilos en las 10 neuronas
            entrenamientoGPU << <1, NUM_NEURONAS >> > (p, pesosGPU, pesosBiasGPU, entradasGPU, deseadoGPU, errorGPU);
            cudaDeviceSynchronize();  // esperar que el kernel termine 
            
            int errorCPU;
            cudaMemcpy(&errorCPU, errorGPU, sizeof(int), cudaMemcpyDeviceToHost);
            if (errorCPU == 1) {
                bucle = false;  // hubo un error en la salida deseada
            }
        }
        // Terminar si hay muchas epocas
        if (epocas > 10000) {
            std::cerr << "Entrenamiento no se cumplio tras 10000 epocas.\n";
            break;
        }
    }

    // ======================================================================================================
    // RESULTADOS DE PESOS
    float pesosCPU[NUM_NEURONAS * NUM_ENTRADAS];
    cudaMemcpy(pesosCPU, pesosGPU, NUM_NEURONAS * NUM_ENTRADAS * sizeof(float), cudaMemcpyDeviceToHost);

    // imprimir pesos entrenados
    std::cout << "Pesos entrenados en " << epocas << " epocas.\n";
    for (int n = 0; n < NUM_NEURONAS; ++n) {
        std::cout << "Neurona " << n << " = pesos: [ ";
        for (int i = 0; i < NUM_ENTRADAS; ++i) {
            std::cout << pesosCPU[n * NUM_ENTRADAS + i] << " ";
        }
        std::cout << "]\n";
    }
    std::cout << "\n\n";


    // ======================================================================================================
    // TEST CON NUEVAS MATRICES
    const int testIndx = 1;
    float testEntradaCPU[testIndx][NUM_ENTRADAS] = {
        {1,1,1,1,1,
        0,0,0,0,1,
        0,1,1,1,1,
        0,0,0,0,1,
        1,1,1,1,1} };

    float salidasCPU[testIndx * NUM_NEURONAS];
    float* salidasGPU = nullptr;
    float* testEntradaGPU = nullptr;
    cudaMalloc((void**)&salidasGPU, testIndx * NUM_NEURONAS * sizeof(float));
    cudaMalloc((void**)&testEntradaGPU, testIndx * NUM_ENTRADAS * sizeof(float));
    cudaMemcpy(testEntradaGPU, testEntradaCPU, sizeof(testEntradaCPU), cudaMemcpyHostToDevice);

    // calcular salidas los pesos entrenados
    for (int p = 0; p < testIndx; ++p) {
        calcularSumaGPU << <1, NUM_NEURONAS >> > (p, pesosGPU, pesosBiasGPU, testEntradaGPU, salidasGPU);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(salidasCPU, salidasGPU, testIndx * NUM_NEURONAS * sizeof(float), cudaMemcpyDeviceToHost);
    
    // imprimir resultados
    for (int p = 0; p < testIndx; ++p) {
        std::cout << "Patron " << p << " = Salida: [ ";
        for (int i = 0; i < NUM_NEURONAS; ++i) {
            std::cout << salidasCPU[p * NUM_NEURONAS + i] << " ";
        }
        std::cout << "]\n";
    }

    // liberar memoria
    cudaFree(testEntradaGPU);
    cudaFree(entradasGPU);
    cudaFree(deseadoGPU);
    cudaFree(pesosGPU);
    cudaFree(pesosBiasGPU);
    cudaFree(errorGPU);
    cudaFree(salidasGPU);

    return 0;
}
