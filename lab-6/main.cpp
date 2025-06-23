#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#define TASA_APRENDIZAJE 0.01
#define EPOCAS 1000
#define TAMANIO_MINI_LOTE 32
#define TAMANIO_ENTRADA 784
#define TAMANIO_SALIDA 10
#define NUM_CAPAS_OCULTAS 120
#define NEURONAS_POR_OCULTA 20

using namespace std;

inline double sigmoide(double x) {
    return 1.0 / (1.0 + exp(-x));
}

inline double derivada_sigmoide(double x) {
    double s = sigmoide(x);
    return s * (1.0 - s);
}

class Matriz {
public:
    vector<vector<double>> datos;
    int filas, columnas;

    Matriz(int r, int c) : filas(r), columnas(c) {
        datos.resize(r, vector<double>(c, 0.0));
    }

    Matriz(const Matriz& otra) : filas(otra.filas), columnas(otra.columnas) {
        datos = otra.datos;
    }

    Matriz& operator=(const Matriz& otra) {
        if (this != &otra) {
            filas = otra.filas;
            columnas = otra.columnas;
            datos = otra.datos;
        }
        return *this;
    }

    void inicializar_aleatorio() {
        random_device rd;
        mt19937 gen(rd());
        double limite = sqrt(6.0 / (filas + columnas));
        uniform_real_distribution<double> dist(-limite, limite);

        #pragma omp parallel for collapse(2)
        for(int i = 0; i < filas; i++) {
            for(int j = 0; j < columnas; j++) {
                datos[i][j] = dist(gen);
            }
        }
    }

    Matriz multiplicar(const Matriz& otra) const {
        Matriz resultado(filas, otra.columnas);

        #pragma omp parallel for collapse(2)
        for(int i = 0; i < filas; i++) {
            for(int j = 0; j < otra.columnas; j++) {
                double suma = 0.0;
                for(int k = 0; k < columnas; k++) {
                    suma += datos[i][k] * otra.datos[k][j];
                }
                resultado.datos[i][j] = suma;
            }
        }
        return resultado;
    }

    Matriz sumar(const Matriz& otra) const {
        Matriz resultado(filas, columnas);

        #pragma omp parallel for collapse(2)
        for(int i = 0; i < filas; i++) {
            for(int j = 0; j < columnas; j++) {
                resultado.datos[i][j] = datos[i][j] + otra.datos[i][j];
            }
        }
        return resultado;
    }

    Matriz restar(const Matriz& otra) const {
        Matriz resultado(filas, columnas);

        #pragma omp parallel for collapse(2)
        for(int i = 0; i < filas; i++) {
            for(int j = 0; j < columnas; j++) {
                resultado.datos[i][j] = datos[i][j] - otra.datos[i][j];
            }
        }
        return resultado;
    }

    Matriz transponer() const {
        Matriz resultado(columnas, filas);

        #pragma omp parallel for collapse(2)
        for(int i = 0; i < filas; i++) {
            for(int j = 0; j < columnas; j++) {
                resultado.datos[j][i] = datos[i][j];
            }
        }
        return resultado;
    }

    Matriz hadamard(const Matriz& otra) const {
        Matriz resultado(filas, columnas);

        #pragma omp parallel for collapse(2)
        for(int i = 0; i < filas; i++) {
            for(int j = 0; j < columnas; j++) {
                resultado.datos[i][j] = datos[i][j] * otra.datos[i][j];
            }
        }
        return resultado;
    }

    Matriz aplicar_funcion(double (*func)(double)) const {
        Matriz resultado(filas, columnas);

        #pragma omp parallel for collapse(2)
        for(int i = 0; i < filas; i++) {
            for(int j = 0; j < columnas; j++) {
                resultado.datos[i][j] = func(datos[i][j]);
            }
        }
        return resultado;
    }

    Matriz multiplicar_escalar(double escalar) const {
        Matriz resultado(filas, columnas);

        #pragma omp parallel for collapse(2)
        for(int i = 0; i < filas; i++) {
            for(int j = 0; j < columnas; j++) {
                resultado.datos[i][j] = datos[i][j] * escalar;
            }
        }
        return resultado;
    }

    void llenar(double valor) {
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < filas; i++) {
            for(int j = 0; j < columnas; j++) {
                datos[i][j] = valor;
            }
        }
    }
};

vector<vector<double>> leer_csv_imagenes(const string& nombre_archivo) {
    ifstream archivo(nombre_archivo);
    if (!archivo.is_open()) {
        cout << "Error: No se puede abrir " << nombre_archivo << endl;
        cout << "Asegúrate de que el archivo existe en el directorio actual." << endl;
        exit(1);
    }

    vector<vector<double>> imagenes;
    string linea;

    while(getline(archivo, linea)) {
        if(linea.empty()) continue;

        vector<double> fila;
        stringstream ss(linea);
        string celda;

        while(getline(ss, celda, ',')) {
            try {
                double valor = stod(celda);
                fila.push_back(valor / 255.0);
            } catch(const exception& e) {
                fila.push_back(0.0);
            }
        }
        if(fila.size() == TAMANIO_ENTRADA + 1) {
            imagenes.push_back(fila);
        }
    }
    archivo.close();
    return imagenes;
}

void separar_datos_etiquetas(const vector<vector<double>>& conjunto_datos,
                             vector<vector<double>>& X_datos,
                             vector<int>& y_datos) {
    X_datos.clear();
    y_datos.clear();

    for(const auto& muestra : conjunto_datos) {
        if(muestra.size() == TAMANIO_ENTRADA + 1) {
            y_datos.push_back((int)muestra[0]);
            vector<double> pixeles(muestra.begin() + 1, muestra.end());
            X_datos.push_back(pixeles);
        }
    }
}

Matriz etiqueta_a_onehot(int etiqueta) {
    Matriz onehot(1, TAMANIO_SALIDA);
    onehot.llenar(0.0);
    if(etiqueta >= 0 && etiqueta < TAMANIO_SALIDA) {
        onehot.datos[0][etiqueta] = 1.0;
    }
    return onehot;
}

void mezclar_datos(vector<vector<double>>& X_datos, vector<int>& y_datos) {
    vector<int> indices(X_datos.size());
    iota(indices.begin(), indices.end(), 0);

    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    vector<vector<double>> X_mezclado(X_datos.size());
    vector<int> y_mezclado(y_datos.size());

    #pragma omp parallel for
    for(int i = 0; i < indices.size(); i++) {
        X_mezclado[i] = X_datos[indices[i]];
        y_mezclado[i] = y_datos[indices[i]];
    }

    X_datos = move(X_mezclado);
    y_datos = move(y_mezclado);
}

class RedNeuronal {
private:
    vector<Matriz> pesos;
    vector<Matriz> sesgos;
    vector<int> tamanio_capas;

public:
    RedNeuronal() {
        tamanio_capas.push_back(TAMANIO_ENTRADA);
        for(int i = 0; i < NUM_CAPAS_OCULTAS; i++) {
            tamanio_capas.push_back(NEURONAS_POR_OCULTA);
        }
        tamanio_capas.push_back(TAMANIO_SALIDA);

        for(int i = 0; i < tamanio_capas.size() - 1; i++) {
            pesos.emplace_back(tamanio_capas[i], tamanio_capas[i + 1]);
            sesgos.emplace_back(1, tamanio_capas[i + 1]);

            pesos[i].inicializar_aleatorio();
            sesgos[i].inicializar_aleatorio();
        }
    }

    

    vector<Matriz> propagar_hacia_adelante(const Matriz& X) {
        vector<Matriz> activaciones;
        vector<Matriz> valores_z;

        Matriz entrada_actual = X;
        activaciones.push_back(entrada_actual);

        for(int i = 0; i < pesos.size(); i++) {
            Matriz z = entrada_actual.multiplicar(pesos[i]).sumar(sesgos[i]);
            valores_z.push_back(z);

            Matriz a = z.aplicar_funcion(sigmoide);
            activaciones.push_back(a);
            entrada_actual = a;
        }

        vector<Matriz> resultados;
        for(const auto& z : valores_z) {
            resultados.push_back(z);
        }
        for(const auto& a : activaciones) {
            resultados.push_back(a);
        }
        return resultados;
    }

    
    
    void propagar_hacia_atras(const Matriz& X, const Matriz& objetivo, const vector<Matriz>& resultados_prop_adelante) {
        int num_capas = pesos.size();
        vector<Matriz> dW(num_capas, Matriz(1, 1));
        vector<Matriz> dB(num_capas, Matriz(1, 1));

        vector<Matriz> valores_z, activaciones;
        for(int i = 0; i < num_capas; i++) {
            valores_z.push_back(resultados_prop_adelante[i]);
        }
        for(int i = 0; i <= num_capas; i++) {
            activaciones.push_back(resultados_prop_adelante[num_capas + i]);
        }

        Matriz delta = activaciones[num_capas].restar(objetivo);

        for(int capa = num_capas - 1; capa >= 0; capa--) {
            dW[capa] = activaciones[capa].transponer().multiplicar(delta);
            dB[capa] = delta;

            if(capa > 0) {
                Matriz delta_peso = delta.multiplicar(pesos[capa].transponer());
                Matriz derivada_sig = valores_z[capa - 1].aplicar_funcion(derivada_sigmoide);
                delta = delta_peso.hadamard(derivada_sig);
            }
        }

        #pragma omp parallel for
        for(int i = 0; i < num_capas; i++) {
            pesos[i] = pesos[i].restar(dW[i].multiplicar_escalar(TASA_APRENDIZAJE));
            sesgos[i] = sesgos[i].restar(dB[i].multiplicar_escalar(TASA_APRENDIZAJE));
        }
    }

    int predecir(const Matriz& X) {
        vector<Matriz> resultados = propagar_hacia_adelante(X);
        const Matriz& salida = resultados[pesos.size() + pesos.size()];

        int indice_max = 0;
        double valor_max = salida.datos[0][0];

        for(int i = 1; i < TAMANIO_SALIDA; i++) {
            if(salida.datos[0][i] > valor_max) {
                valor_max = salida.datos[0][i];
                indice_max = i;
            }
        }
        return indice_max;
    }

    double calcular_error_epoca(const vector<vector<double>>& X_datos, const vector<int>& y_datos) {
        double error_total = 0.0;

        #pragma omp parallel for reduction(+:error_total)
        for(int i = 0; i < X_datos.size(); i++) {
            Matriz X(1, TAMANIO_ENTRADA);
            for(int j = 0; j < TAMANIO_ENTRADA; j++) {
                X.datos[0][j] = X_datos[i][j];
            }

            Matriz objetivo = etiqueta_a_onehot(y_datos[i]);
            vector<Matriz> resultados = propagar_hacia_adelante(X);
            const Matriz& salida = resultados[pesos.size() + pesos.size()];

            double error_muestra = 0.0;
            for(int j = 0; j < TAMANIO_SALIDA; j++) {
                double diff = salida.datos[0][j] - objetivo.datos[0][j];
                error_muestra += diff * diff;
            }
            error_total += error_muestra;
        }
        return error_total;
    }

    double calcular_precision(const vector<vector<double>>& X_datos, const vector<int>& y_datos, int max_muestras = -1) {
        int muestras = (max_muestras > 0) ? min(max_muestras, (int)X_datos.size()) : X_datos.size();
        int correctos = 0;

        #pragma omp parallel for reduction(+:correctos)
        for(int i = 0; i < muestras; i++) {
            Matriz X(1, TAMANIO_ENTRADA);
            for(int j = 0; j < TAMANIO_ENTRADA; j++) {
                X.datos[0][j] = X_datos[i][j];
            }

            int predicho = predecir(X);
            if(predicho == y_datos[i]) {
                correctos++;
            }
        }
        return (double)correctos / muestras * 100.0;
    }

    void imprimir_matriz_confusion(const vector<vector<double>>& X_datos, const vector<int>& y_datos, int max_muestras = 1000) {
        int muestras = min(max_muestras, (int)X_datos.size());
        vector<vector<int>> confusion(TAMANIO_SALIDA, vector<int>(TAMANIO_SALIDA, 0));

        #pragma omp parallel for collapse(2)
        
        for(int i = 0; i < TAMANIO_SALIDA; ++i) {
            for(int j = 0; j < TAMANIO_SALIDA; ++j) {
                confusion[i][j] = 0;
            }
        }

        
        for(int i = 0; i < muestras; i++) {
            Matriz X(1, TAMANIO_ENTRADA);
            for(int j = 0; j < TAMANIO_ENTRADA; j++) {
                X.datos[0][j] = X_datos[i][j];
            }

            int predicho = predecir(X);
            if(y_datos[i] >= 0 && y_datos[i] < TAMANIO_SALIDA && predicho >= 0 && predicho < TAMANIO_SALIDA) {
                #pragma omp atomic update
                confusion[y_datos[i]][predicho]++;
            }
        }

        cout << "\nMatriz de Confusión (muestras: " << muestras << "):" << endl;
        cout << "    ";
        for(int i = 0; i < TAMANIO_SALIDA; i++) {
            cout << setw(6) << i;
        }
        cout << endl;

        for(int i = 0; i < TAMANIO_SALIDA; i++) {
            cout << setw(3) << i << ": ";
            for(int j = 0; j < TAMANIO_SALIDA; j++) {
                cout << setw(6) << confusion[i][j];
            }
            cout << endl;
        }
    }

    void entrenar_lote(const vector<vector<double>>& X_lote, const vector<int>& y_lote) {
        #pragma omp parallel for
        for(int i = 0; i < X_lote.size(); i++) {
            Matriz X(1, TAMANIO_ENTRADA);
            for(int j = 0; j < TAMANIO_ENTRADA; j++) {
                X.datos[0][j] = X_lote[i][j];
            }

            Matriz objetivo = etiqueta_a_onehot(y_lote[i]);
            vector<Matriz> resultados_prop_adelante = propagar_hacia_adelante(X);
            
            propagar_hacia_atras(X, objetivo, resultados_prop_adelante);
        }
    }
};

int main() {
    #ifdef _OPENMP
    omp_set_num_threads(omp_get_max_threads());
    #endif

    cout << "=== Red Neuronal MNIST Optimizada con OpenMP ===" << endl;
    cout << "Configuración:" << endl;
    cout << "- Tasa de aprendizaje: " << TASA_APRENDIZAJE << endl;
    cout << "- Épocas: " << EPOCAS << endl;
    cout << "- Tamaño de mini-lote: " << TAMANIO_MINI_LOTE << endl;
    cout << "- Capas ocultas: " << NUM_CAPAS_OCULTAS << endl;
    cout << "- Neuronas por capa oculta: " << NEURONAS_POR_OCULTA << endl;
    #ifdef _OPENMP
    cout << "- Hilos OpenMP: " << omp_get_max_threads() << endl;
    #else
    cout << "- OpenMP no habilitado." << endl;
    #endif
    cout << "================================" << endl;

    cout << "\nCargando datos MNIST..." << endl;
    cout << "Formato esperado: CSV con etiqueta en primera columna y " << TAMANIO_ENTRADA << " píxeles." << endl;

    vector<string> archivos_entrenamiento = {"mnist_train.csv", "train.csv", "mnist_train.txt", "train.txt"};
    vector<string> archivos_prueba = {"mnist_test.csv", "test.csv", "mnist_test.txt", "test.txt"};

    vector<vector<double>> conjunto_entrenamiento, conjunto_prueba;

    bool entrenamiento_encontrado = false;
    for(const string& nombre_archivo : archivos_entrenamiento) {
        ifstream archivo_temp(nombre_archivo);
        if(archivo_temp.is_open()) {
            archivo_temp.close();
            cout << "Encontrado archivo de entrenamiento: " << nombre_archivo << endl;
            conjunto_entrenamiento = leer_csv_imagenes(nombre_archivo);
            entrenamiento_encontrado = true;
            break;
        }
    }

    if(!entrenamiento_encontrado) {
        cout << "Error: No se encontró ningún archivo de entrenamiento." << endl;
        cout << "Archivos buscados: mnist_train.csv, train.csv, mnist_train.txt, train.txt" << endl;
        return 1;
    }

    bool prueba_encontrada = false;
    for(const string& nombre_archivo : archivos_prueba) {
        ifstream archivo_temp(nombre_archivo);
        if(archivo_temp.is_open()) {
            archivo_temp.close();
            cout << "Encontrado archivo de prueba: " << nombre_archivo << endl;
            conjunto_prueba = leer_csv_imagenes(nombre_archivo);
            prueba_encontrada = true;
            break;
        }
    }

    if(!prueba_encontrada) {
        cout << "Advertencia: No se encontró archivo de prueba. Usando parte del entrenamiento para prueba." << endl;
        int punto_division = max(0, (int)conjunto_entrenamiento.size() - 10000);
        conjunto_prueba.assign(conjunto_entrenamiento.begin() + punto_division, conjunto_entrenamiento.end());
        conjunto_entrenamiento.resize(punto_division);
    }

    vector<vector<double>> X_entrenamiento, X_prueba;
    vector<int> y_entrenamiento, y_prueba;

    separar_datos_etiquetas(conjunto_entrenamiento, X_entrenamiento, y_entrenamiento);
    separar_datos_etiquetas(conjunto_prueba, X_prueba, y_prueba);

    cout << "Datos procesados:" << endl;
    cout << "- Entrenamiento: " << X_entrenamiento.size() << " muestras" << endl;
    cout << "- Prueba: " << X_prueba.size() << " muestras" << endl;

    if(X_entrenamiento.empty()) {
        cout << "Error: No se pudieron cargar datos de entrenamiento válidos." << endl;
        return 1;
    }

    RedNeuronal rn;

    cout << "\nIniciando entrenamiento..." << endl;
    auto inicio_total = chrono::high_resolution_clock::now();

    for(int epoca = 0; epoca < EPOCAS; epoca++) {
        auto inicio_epoca = chrono::high_resolution_clock::now();

        mezclar_datos(X_entrenamiento, y_entrenamiento);

        int num_lotes = (X_entrenamiento.size() + TAMANIO_MINI_LOTE - 1) / TAMANIO_MINI_LOTE;

        for(int lote = 0; lote < num_lotes; lote++) {
            int inicio_idx = lote * TAMANIO_MINI_LOTE;
            int fin_idx = min(inicio_idx + TAMANIO_MINI_LOTE, (int)X_entrenamiento.size());

            vector<vector<double>> X_lote(X_entrenamiento.begin() + inicio_idx, X_entrenamiento.begin() + fin_idx);
            vector<int> y_lote(y_entrenamiento.begin() + inicio_idx, y_entrenamiento.begin() + fin_idx);

            rn.entrenar_lote(X_lote, y_lote);
        }

        auto fin_epoca = chrono::high_resolution_clock::now();
        auto duracion_epoca = chrono::duration_cast<chrono::milliseconds>(fin_epoca - inicio_epoca);

        if((epoca + 1) % 10 == 0) {
            double error_epoca = rn.calcular_error_epoca(X_entrenamiento, y_entrenamiento);
            double precision_entrenamiento = rn.calcular_precision(X_entrenamiento, y_entrenamiento, min(5000, (int)X_entrenamiento.size()));
            double precision_prueba = rn.calcular_precision(X_prueba, y_prueba, min(1000, (int)X_prueba.size()));

            cout << "Época " << setw(4) << (epoca + 1) << "/" << EPOCAS
                 << " | Tiempo: " << setw(4) << duracion_epoca.count() << "ms"
                 << " | Error Total: " << fixed << setprecision(2) << error_epoca
                 << " | Entr.: " << fixed << setprecision(2) << precision_entrenamiento << "%"
                 << " | Prueba: " << fixed << setprecision(2) << precision_prueba << "%" << endl;
        }

        if((epoca + 1) % 100 == 0 && !X_prueba.empty()) {
            rn.imprimir_matriz_confusion(X_prueba, y_prueba, 1000);
        }
    }

    auto fin_total = chrono::high_resolution_clock::now();
    auto duracion_total = chrono::duration_cast<chrono::seconds>(fin_total - inicio_total);

    cout << "\n=== EVALUACIÓN FINAL ===" << endl;
    cout << "Tiempo total de entrenamiento: " << duracion_total.count() << " segundos" << endl;

    double precision_entrenamiento_final = rn.calcular_precision(X_entrenamiento, y_entrenamiento);
    cout << "Precisión final en entrenamiento: " << fixed << setprecision(2) << precision_entrenamiento_final << "%" << endl;

    if(!X_prueba.empty()) {
        double precision_prueba_final = rn.calcular_precision(X_prueba, y_prueba);
        cout << "Precisión final en prueba: " << fixed << setprecision(2) << precision_prueba_final << "%" << endl;

        rn.imprimir_matriz_confusion(X_prueba, y_prueba, 2000);
    }

    cout << "\n¡Entrenamiento completado exitosamente!" << endl;

    return 0;
}
