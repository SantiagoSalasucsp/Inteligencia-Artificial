#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <omp.h>

using namespace std;

#define NUM_EPOCAS 50
#define TAMANO_MINI_LOTE 32
#define NUM_CAPAS_OCULTAS 2
#define NEURONAS_POR_CAPA_OCULTA 128
#define TASA_APRENDIZAJE 0.1
#define TAMANO_ENTRADA 784
#define TAMANO_SALIDA 10

class NeuralNetwork {
private:
    vector<vector<vector<double>>> pesos;
    vector<vector<double>> bias;
    vector<vector<double>> activaciones;
    vector<vector<double>> valores_z;
    
    vector<vector<vector<double>>> gradientes_pesos;
    vector<vector<double>> gradientes_bias;
    
    mt19937 gen;
    
public:
    NeuralNetwork() : gen(42) {
        inicializarRed();
    }
    
    void inicializarRed() {
        vector<int> tamanos_capa;
        tamanos_capa.push_back(TAMANO_ENTRADA);
        for(int i = 0; i < NUM_CAPAS_OCULTAS; i++) {
            tamanos_capa.push_back(NEURONAS_POR_CAPA_OCULTA);
        }
        tamanos_capa.push_back(TAMANO_SALIDA);
        
        int num_capas = tamanos_capa.size() - 1;
        pesos.resize(num_capas);
        bias.resize(num_capas);
        gradientes_pesos.resize(num_capas);
        gradientes_bias.resize(num_capas);
        
        for(int i = 0; i < num_capas; i++) {
            double escala = sqrt(2.0 / (tamanos_capa[i] + tamanos_capa[i + 1]));
            normal_distribution<double> dist_pesos(0.0, escala);
            normal_distribution<double> dist_bias(0.0, 0.01);
            
            pesos[i].assign(tamanos_capa[i], vector<double>(tamanos_capa[i + 1]));
            gradientes_pesos[i].assign(tamanos_capa[i], vector<double>(tamanos_capa[i + 1], 0.0));
            
            for(int j = 0; j < tamanos_capa[i]; j++) {
                for(int k = 0; k < tamanos_capa[i + 1]; k++) {
                    pesos[i][j][k] = dist_pesos(gen);
                }
            }
            
            bias[i].assign(tamanos_capa[i + 1], 0.0);
            gradientes_bias[i].assign(tamanos_capa[i + 1], 0.0);
            for(int j = 0; j < tamanos_capa[i + 1]; j++) {
                bias[i][j] = dist_bias(gen);
            }
        }
        
        activaciones.resize(tamanos_capa.size());
        valores_z.resize(num_capas);
        
        for(int i = 0; i < tamanos_capa.size(); i++) {
            activaciones[i].resize(tamanos_capa[i]);
        }
        
        for(int i = 0; i < num_capas; i++) {
            valores_z[i].resize(tamanos_capa[i + 1]);
        }
    }
    
    double sigmoide(double x) {
        if(x > 500) return 1.0;
        if(x < -500) return 0.0;
        return 1.0 / (1.0 + exp(-x));
    }
    
    double derivadaSigmoide(double valor_activado) {
        return valor_activado * (1.0 - valor_activado);
    }
    
    vector<double> propagacionHaciaAdelante(const vector<double>& entrada) {
        for(int i = 0; i < entrada.size(); i++) {
            activaciones[0][i] = entrada[i];
        }
        
        for(int capa = 0; capa < pesos.size(); capa++) {
            int tamano_entrada = activaciones[capa].size();
            int tamano_salida = valores_z[capa].size();
            
            #pragma omp parallel for schedule(static)
            for(int j = 0; j < tamano_salida; j++) {
                double suma = bias[capa][j];
                for(int i = 0; i < tamano_entrada; i++) {
                    suma += activaciones[capa][i] * pesos[capa][i][j];
                }
                valores_z[capa][j] = suma;
                activaciones[capa + 1][j] = sigmoide(suma);
            }
        }
        
        return activaciones.back();
    }
    
    void propagacionHaciaAtras(const vector<double>& objetivo) {
        int num_capas = pesos.size();
        vector<vector<double>> deltas(num_capas);
        
        for(int i = 0; i < num_capas; i++) {
            deltas[i].resize(activaciones[i + 1].size());
        }
        
        int capa_salida = num_capas - 1;
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < activaciones.back().size(); i++) {
            double error = activaciones.back()[i] - objetivo[i];
            deltas[capa_salida][i] = error * derivadaSigmoide(activaciones.back()[i]);
        }
        
        for(int capa = num_capas - 2; capa >= 0; capa--) {
            int tamano_actual = deltas[capa].size();
            int tamano_siguiente = deltas[capa + 1].size();
            
            #pragma omp parallel for schedule(static)
            for(int i = 0; i < tamano_actual; i++) {
                double suma = 0.0;
                for(int j = 0; j < tamano_siguiente; j++) {
                    suma += deltas[capa + 1][j] * pesos[capa + 1][i][j];
                }
                deltas[capa][i] = suma * derivadaSigmoide(activaciones[capa + 1][i]);
            }
        }
        
        for(int capa = 0; capa < num_capas; capa++) {
            int tamano_entrada = activaciones[capa].size();
            int tamano_salida = deltas[capa].size();
            
            for(int i = 0; i < tamano_entrada; i++) {
                for(int j = 0; j < tamano_salida; j++) {
                    gradientes_pesos[capa][i][j] += activaciones[capa][i] * deltas[capa][j];
                }
            }
            
            for(int j = 0; j < tamano_salida; j++) {
                gradientes_bias[capa][j] += deltas[capa][j];
            }
        }
    }
    
    void actualizarPesos(int tamano_lote) {
        double lr = TASA_APRENDIZAJE / tamano_lote;
        
        for(int capa = 0; capa < pesos.size(); capa++) {
            int tamano_entrada = pesos[capa].size();
            int tamano_salida = pesos[capa][0].size();
            
            #pragma omp parallel for schedule(static) collapse(2)
            for(int i = 0; i < tamano_entrada; i++) {
                for(int j = 0; j < tamano_salida; j++) {
                    pesos[capa][i][j] -= lr * gradientes_pesos[capa][i][j];
                    gradientes_pesos[capa][i][j] = 0.0;
                }
            }
            
            #pragma omp parallel for schedule(static)
            for(int j = 0; j < tamano_salida; j++) {
                bias[capa][j] -= lr * gradientes_bias[capa][j];
                gradientes_bias[capa][j] = 0.0;
            }
        }
    }
    
    double calcularError(const vector<double>& objetivo) {
        double error = 0.0;
        
        for(int i = 0; i < activaciones.back().size(); i++) {
            double diff = activaciones.back()[i] - objetivo[i];
            error += diff * diff;
        }
        
        return error * 0.5;
    }
    
    int predecir(const vector<double>& entrada) {
        vector<double> salida = propagacionHaciaAdelante(entrada);
        return max_element(salida.begin(), salida.end()) - salida.begin();
    }
};

vector<double> codificacionOneHot(int etiqueta) {
    vector<double> codificado(TAMANO_SALIDA, 0.0);
    if(etiqueta >= 0 && etiqueta < TAMANO_SALIDA) {
        codificado[etiqueta] = 1.0;
    }
    return codificado;
}

pair<vector<vector<double>>, vector<int>> cargarDatos(const string& nombre_archivo) {
    vector<vector<double>> datos;
    vector<int> etiquetas;
    
    ifstream archivo(nombre_archivo);
    if(!archivo.is_open()) {
        cout << "Error: No se pudo abrir " << nombre_archivo << endl;
        return make_pair(datos, etiquetas);
    }
    
    string linea;
    while(getline(archivo, linea)) {
        if(linea.empty()) continue;
        
        stringstream ss(linea);
        string celda;
        
        vector<double> fila;
        bool primera = true;
        int etiqueta = 0;
        
        while(getline(ss, celda, ',')) {
            if(primera) {
                etiqueta = stoi(celda);
                primera = false;
            } else {
                double valor_pixel = stod(celda) / 255.0;
                fila.push_back(valor_pixel);
            }
        }
        
        if(fila.size() == TAMANO_ENTRADA) {
            datos.push_back(fila);
            etiquetas.push_back(etiqueta);
        }
    }
    
    archivo.close();
    return make_pair(datos, etiquetas);
}

void imprimirMatrizConfusion(const vector<vector<int>>& matriz) {
    cout << "\nMatriz de Confusion:\n";
    cout << "    ";
    for(int i = 0; i < 10; i++) {
        cout << setw(6) << i;
    }
    cout << "\n";
    
    for(int i = 0; i < 10; i++) {
        cout << setw(2) << i << ": ";
        for(int j = 0; j < 10; j++) {
            cout << setw(6) << matriz[i][j];
        }
        cout << "\n";
    }
    
    int correcto = 0, total = 0;
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            if(i == j) correcto += matriz[i][j];
            total += matriz[i][j];
        }
    }
    
    cout << "\nPrecision total: " << fixed << setprecision(2)
            << (100.0 * correcto / total) << "%\n";
    
    cout << "\nPrecision por clase:\n";
    for(int i = 0; i < 10; i++) {
        int total_clase = 0;
        for(int j = 0; j < 10; j++) {
            total_clase += matriz[i][j];
        }
        if(total_clase > 0) {
            double precision_clase = 100.0 * matriz[i][i] / total_clase;
            cout << "Clase " << i << ": " << fixed << setprecision(2)
                    << precision_clase << "%\n";
        }
    }
}

vector<int> crearIndicesAleatorios(int tamano) {
    vector<int> indices(tamano);
    for(int i = 0; i < tamano; i++) {
        indices[i] = i;
    }
    
    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);
    
    return indices;
}

int main() {
    cout << "Inicializando Red Neuronal...\n";
    cout << "Arquitectura: " << TAMANO_ENTRADA << " -> ";
    for(int i = 0; i < NUM_CAPAS_OCULTAS; i++) {
        cout << NEURONAS_POR_CAPA_OCULTA << " -> ";
    }
    cout << TAMANO_SALIDA << "\n";
    cout << "Epocas: " << NUM_EPOCAS << ", Mini-batch: " << TAMANO_MINI_LOTE
            << ", LR: " << TASA_APRENDIZAJE << "\n\n";
    
    NeuralNetwork nn;
    
    string ruta_train = "/Users/santiagosalas/Desktop/mi-repositorio/percepCPU New/prueba omp/prueba omp/mnist_train.csv";
    string ruta_test = "/Users/santiagosalas/Desktop/mi-repositorio/percepCPU New/prueba omp/prueba omp/mnist_test.csv";
    
    cout << "Cargando datos...\n";
    auto datos_entrenamiento = cargarDatos(ruta_train);
    auto datos_prueba = cargarDatos(ruta_test);
    
    cout << "Datos de entrenamiento: " << datos_entrenamiento.first.size() << " muestras\n";
    cout << "Datos de prueba: " << datos_prueba.first.size() << " muestras\n\n";
    
    if(datos_entrenamiento.first.empty() || datos_prueba.first.empty()) {
        cout << "Error cargando datos. Verifica las rutas.\n";
        return 1;
    }
    
    cout << "Iniciando entrenamiento...\n";
    vector<double> errores_epoca;
    
    auto tiempo_inicio = chrono::high_resolution_clock::now();
    
    for(int epoca = 0; epoca < NUM_EPOCAS; epoca++) {
        double error_total = 0.0;
        vector<int> indices = crearIndicesAleatorios(datos_entrenamiento.first.size());
        
        int num_lotes = (datos_entrenamiento.first.size() + TAMANO_MINI_LOTE - 1) / TAMANO_MINI_LOTE;
        
        for(int lote = 0; lote < num_lotes; lote++) {
            int indice_inicio = lote * TAMANO_MINI_LOTE;
            int indice_fin = min(indice_inicio + TAMANO_MINI_LOTE, (int)datos_entrenamiento.first.size());
            
            for(int i = indice_inicio; i < indice_fin; i++) {
                int idx = indices[i];
                vector<double> objetivo = codificacionOneHot(datos_entrenamiento.second[idx]);
                
                nn.propagacionHaciaAdelante(datos_entrenamiento.first[idx]);
                error_total += nn.calcularError(objetivo);
                nn.propagacionHaciaAtras(objetivo);
            }
            
            nn.actualizarPesos(indice_fin - indice_inicio);
        }
        
        errores_epoca.push_back(error_total / datos_entrenamiento.first.size());
        
        if((epoca + 1) % 10 == 0 || epoca == 0) {
            int inicio_promedio = max(0, (int)errores_epoca.size() - 10);
            double error_promedio = 0.0;
            for(int i = inicio_promedio; i < errores_epoca.size(); i++) {
                error_promedio += errores_epoca[i];
            }
            error_promedio /= (errores_epoca.size() - inicio_promedio);
            
            auto tiempo_actual = chrono::high_resolution_clock::now();
            auto transcurrido = chrono::duration_cast<chrono::seconds>(tiempo_actual - tiempo_inicio);
            
            cout << "Epoca " << (epoca + 1) << ", Error promedio: "
                    << fixed << setprecision(6) << error_promedio
                    << ", Tiempo: " << transcurrido.count() << "s\n";
        }
    }
    
    cout << "\n== TESTING ==\n";
    
    vector<vector<int>> matriz_confusion(10, vector<int>(10, 0));
    
    for(int i = 0; i < datos_prueba.first.size(); i++) {
        int predicho = nn.predecir(datos_prueba.first[i]);
        int real = datos_prueba.second[i];
        
        if(predicho >= 0 && predicho < 10 && real >= 0 && real < 10) {
            matriz_confusion[real][predicho]++;
        }
    }
    
    imprimirMatrizConfusion(matriz_confusion);
    
    return 0;
}
