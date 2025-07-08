#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <omp.h>
#define NUM_EPOCHS 20
#define SAMPLES_PER_EPOCH 10000
using namespace std;

vector<vector<int>> datos_train;
vector<vector<int>> datos_test;
bool datos_cargados = false;

vector<vector<vector<float>>> W_Q;
vector<vector<vector<float>>> W_K;
vector<vector<vector<float>>> W_V;

void init_Q_K_V(int num_heads, int dim_model, int dim_head) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> d(0.0f, 0.02f);

    W_Q.resize(num_heads, vector<vector<float>>(dim_model, vector<float>(dim_head)));
    W_K.resize(num_heads, vector<vector<float>>(dim_model, vector<float>(dim_head)));
    W_V.resize(num_heads, vector<vector<float>>(dim_model, vector<float>(dim_head)));

    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < dim_model; ++i) {
            for (int j = 0; j < dim_head; ++j) {
                W_Q[h][i][j] = d(gen);
                W_K[h][i][j] = d(gen);
                W_V[h][i][j] = d(gen);
            }
        }
    }
}

void cargar_datos() {
    if (datos_cargados) {
        return;
    }

    string train_file = "mnist_train.csv";
    string test_file = "mnist_test.csv";

    auto read_csv = [](const string& filename, vector<vector<int>>& data) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: No se pudo abrir el archivo " << filename << endl;
            return;
        }
        
        string line;
        while (getline(file, line)) {
            if (line.empty()) continue;
            
            stringstream ss(line);
            string cell;
            vector<int> row;
            
            while (getline(ss, cell, ',')) {
                try {
                    row.push_back(stoi(cell));
                } catch (const exception& e) {
                    cerr << "Error al convertir: " << cell << endl;
                    continue;
                }
            }
            
            if (!row.empty()) {
                data.push_back(row);
            }
        }
        file.close();
    };

    read_csv(train_file, datos_train);
    read_csv(test_file, datos_test);

    datos_cargados = true;
    cout << "Datos cargados: " << datos_train.size() << " muestras de entrenamiento, "
         << datos_test.size() << " muestras de test." << endl;
}

vector<vector<int>> obtener_parches(int indice, bool es_train = true) {
    if (!datos_cargados) {
        cargar_datos();
    }

    const vector<vector<int>>& dataset = es_train ? datos_train : datos_test;
    
    if (indice >= static_cast<int>(dataset.size()) || indice < 0) {
        cerr << "Error: Índice fuera de rango" << endl;
        return {};
    }
    
    if (dataset[indice].size() < 785) {
        cerr << "Error: Datos insuficientes en la muestra" << endl;
        return {};
    }

    vector<int> imagen_plana;
    for (size_t i = 1; i < dataset[indice].size(); ++i) {
        imagen_plana.push_back(dataset[indice][i]);
    }

    vector<vector<int>> imagen_28x28(28, vector<int>(28));
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            imagen_28x28[i][j] = imagen_plana[i * 28 + j];
        }
    }

    vector<vector<int>> parches;
    int patch_size = 7;
    for (int i = 0; i < 28 / patch_size; ++i) {
        for (int j = 0; j < 28 / patch_size; ++j) {
            vector<int> current_patch;
            for (int row = i * patch_size; row < (i + 1) * patch_size; ++row) {
                for (int col = j * patch_size; col < (j + 1) * patch_size; ++col) {
                    current_patch.push_back(imagen_28x28[row][col]);
                }
            }
            parches.push_back(current_patch);
        }
    }
    return parches;
}

int obtener_etiqueta(int indice, bool es_train = true) {
    if (!datos_cargados) {
        cargar_datos();
    }

    const vector<vector<int>>& dataset = es_train ? datos_train : datos_test;
    
    if (indice >= static_cast<int>(dataset.size()) || indice < 0) {
        cerr << "Error: Índice fuera de rango" << endl;
        return -1;
    }
    
    if (dataset[indice].empty()) {
        cerr << "Error: Datos vacíos en la muestra" << endl;
        return -1;
    }

    return dataset[indice][0];
}

vector<vector<float>> positional_encoding(const vector<vector<int>>& parches) {
    if (parches.empty() || parches[0].empty()) {
        return {};
    }
    
    vector<vector<float>> parches_pe(parches.size(), vector<float>(parches[0].size()));
    int dim = parches[0].size();

    for (size_t i = 0; i < parches.size(); ++i) {
        for (int j = 0; j < dim; ++j) {
            float pe_val;
            if (j % 2 == 0) {
                pe_val = sin(static_cast<float>(i) / pow(10000.0f, static_cast<float>(j) / dim));
            } else {
                pe_val = cos(static_cast<float>(i) / pow(10000.0f, static_cast<float>(j - 1) / dim));
            }
            parches_pe[i][j] = (static_cast<float>(parches[i][j]) / 255.0f) + pe_val;
        }
    }
    return parches_pe;
}

vector<vector<float>> transponer(const vector<vector<float>>& matriz) {
    if (matriz.empty() || matriz[0].empty()) {
        return {};
    }
    
    int filas = matriz.size();
    int cols = matriz[0].size();
    vector<vector<float>> transpuesta(cols, vector<float>(filas));
    
    for (int i = 0; i < filas; ++i) {
        for (int j = 0; j < cols; ++j) {
            transpuesta[j][i] = matriz[i][j];
        }
    }
    return transpuesta;
}

vector<vector<float>> multiplicar_matrices(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    if (A.empty() || B.empty() || A[0].empty() || B[0].empty()) {
        return {};
    }

    int filas_A = A.size();
    int cols_A = A[0].size();
    int filas_B = B.size();
    int cols_B = B[0].size();

    if (cols_A != filas_B) {
        cerr << "Error: Dimensiones incompatibles para multiplicación: "
             << cols_A << " != " << filas_B << endl;
        return {};
    }

    vector<vector<float>> resultado(filas_A, vector<float>(cols_B, 0.0f));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < filas_A; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            for (int k = 0; k < cols_A; ++k) {
                resultado[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return resultado;
}

vector<vector<float>> softmax(const vector<vector<float>>& matriz) {
    if (matriz.empty() || matriz[0].empty()) {
        return {};
    }

    vector<vector<float>> resultado = matriz;
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < resultado.size(); ++i) {
        float max_val = *max_element(resultado[i].begin(), resultado[i].end());

        float sum_exp = 0.0f;
        for (size_t j = 0; j < resultado[i].size(); ++j) {
            resultado[i][j] = exp(resultado[i][j] - max_val);
            sum_exp += resultado[i][j];
        }

        if (sum_exp > 0) {
            for (size_t j = 0; j < resultado[i].size(); ++j) {
                resultado[i][j] /= sum_exp;
            }
        }
    }
    return resultado;
}

vector<vector<float>> attention_head(const vector<vector<float>>& input, int head_idx, int dim_head) {
    if (input.empty() || input[0].empty()) {
        return {};
    }
    
    if (head_idx >= static_cast<int>(W_Q.size()) || head_idx < 0) {
        cerr << "Error: Índice de cabeza de atención fuera de rango" << endl;
        return {};
    }

    vector<vector<float>> Q = multiplicar_matrices(input, W_Q[head_idx]);
    vector<vector<float>> K = multiplicar_matrices(input, W_K[head_idx]);
    vector<vector<float>> V = multiplicar_matrices(input, W_V[head_idx]);

    if (Q.empty() || K.empty() || V.empty()) {
        return {};
    }

    vector<vector<float>> K_T = transponer(K);
    vector<vector<float>> scores = multiplicar_matrices(Q, K_T);

    if (scores.empty()) {
        return {};
    }

    float scale = 1.0f / sqrt(static_cast<float>(dim_head));
    for (size_t i = 0; i < scores.size(); ++i) {
        for (size_t j = 0; j < scores[0].size(); ++j) {
            scores[i][j] *= scale;
        }
    }

    vector<vector<float>> attention_weights = softmax(scores);
    vector<vector<float>> output = multiplicar_matrices(attention_weights, V);

    return output;
}

vector<vector<float>> suma_matrices(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    if (A.empty() || B.empty() || A[0].empty() || B[0].empty()) {
        return {};
    }

    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        cerr << "Error: Dimensiones incompatibles para suma: "
             << A.size() << "x" << A[0].size() << " vs "
             << B.size() << "x" << B[0].size() << endl;
        return {};
    }

    vector<vector<float>> resultado(A.size(), vector<float>(A[0].size()));
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            resultado[i][j] = A[i][j] + B[i][j];
        }
    }
    return resultado;
}

vector<vector<float>> layer_norm(const vector<vector<float>>& input, float epsilon = 1e-6f) {
    if (input.empty() || input[0].empty()) {
        return {};
    }

    vector<vector<float>> normalized_output = input;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < input.size(); ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < input[i].size(); ++j) {
            sum += input[i][j];
        }
        float mean = sum / input[i].size();

        float variance_sum = 0.0f;
        for (size_t j = 0; j < input[i].size(); ++j) {
            float diff = input[i][j] - mean;
            variance_sum += diff * diff;
        }
        float variance = variance_sum / input[i].size();
        float std_dev = sqrt(variance + epsilon);

        for (size_t j = 0; j < input[i].size(); ++j) {
            normalized_output[i][j] = (input[i][j] - mean) / std_dev;
        }
    }
    return normalized_output;
}

vector<double> etiqueta_a_onehot(int etiqueta) {
    vector<double> one_hot(10, 0.0);
    if (etiqueta >= 0 && etiqueta < 10) {
        one_hot[etiqueta] = 1.0;
    }
    return one_hot;
}

struct MLP {
    vector<vector<vector<double>>> pesos;
    vector<vector<double>> bias;
    vector<vector<double>> activaciones;
    vector<vector<double>> valores_z;
    
    vector<vector<vector<double>>> gradientes_pesos;
    vector<vector<double>> gradientes_bias;
    
    bool hacer_backward;
    mt19937 gen;
    
    MLP(int num_entrada, int num_salida, int num_capas_escondidas, int num_neuronas_por_capa_escondida, int si_backward)
        : hacer_backward(si_backward == 1), gen(42) {
        inicializarRed(num_entrada, num_salida, num_capas_escondidas, num_neuronas_por_capa_escondida);
    }
    
    void inicializarRed(int num_entrada, int num_salida, int num_capas_escondidas, int num_neuronas_por_capa_escondida) {
        vector<int> tamanos_capa;
        tamanos_capa.push_back(num_entrada);
        for(int i = 0; i < num_capas_escondidas; i++) {
            tamanos_capa.push_back(num_neuronas_por_capa_escondida);
        }
        tamanos_capa.push_back(num_salida);
        
        int num_capas = tamanos_capa.size() - 1;
        pesos.resize(num_capas);
        bias.resize(num_capas);
        
        if(hacer_backward) {
            gradientes_pesos.resize(num_capas);
            gradientes_bias.resize(num_capas);
        }
        
        for(int i = 0; i < num_capas; i++) {
            double escala = sqrt(2.0 / (tamanos_capa[i] + tamanos_capa[i + 1]));
            normal_distribution<double> dist_pesos(0.0, escala);
            normal_distribution<double> dist_bias(0.0, 0.01);
            
            pesos[i].assign(tamanos_capa[i], vector<double>(tamanos_capa[i + 1]));
            
            if(hacer_backward) {
                gradientes_pesos[i].assign(tamanos_capa[i], vector<double>(tamanos_capa[i + 1], 0.0));
            }
            
            for(int j = 0; j < tamanos_capa[i]; j++) {
                for(int k = 0; k < tamanos_capa[i + 1]; k++) {
                    pesos[i][j][k] = dist_pesos(gen);
                }
            }
            
            bias[i].assign(tamanos_capa[i + 1], 0.0);
            if(hacer_backward) {
                gradientes_bias[i].assign(tamanos_capa[i + 1], 0.0);
            }
        }
        
        activaciones.resize(tamanos_capa.size());
        valores_z.resize(num_capas);
        
        for(size_t i = 0; i < tamanos_capa.size(); i++) {
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
        if(entrada.size() != activaciones[0].size()) {
            cerr << "Error: Tamaño de entrada incorrecto" << endl;
            return {};
        }
        
        for(size_t i = 0; i < entrada.size(); i++) {
            activaciones[0][i] = entrada[i];
        }
        
        for(size_t capa = 0; capa < pesos.size(); capa++) {
            int tamano_entrada = activaciones[capa].size();
            int tamano_salida = valores_z[capa].size();
            
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
        if(!hacer_backward) return;
        
        if(objetivo.size() != activaciones.back().size()) {
            cerr << "Error: Tamaño de objetivo incorrecto" << endl;
            return;
        }
        
        int num_capas = pesos.size();
        vector<vector<double>> deltas(num_capas);
        
        for(int i = 0; i < num_capas; i++) {
            deltas[i].resize(activaciones[i + 1].size());
        }
        
        int capa_salida = num_capas - 1;
        for(size_t i = 0; i < activaciones.back().size(); i++) {
            double error = activaciones.back()[i] - objetivo[i];
            deltas[capa_salida][i] = error * derivadaSigmoide(activaciones.back()[i]);
        }
        
        for(int capa = num_capas - 2; capa >= 0; capa--) {
            int tamano_actual = deltas[capa].size();
            int tamano_siguiente = deltas[capa + 1].size();
            
            for(int i = 0; i < tamano_actual; i++) {
                double suma = 0.0;
                for(int j = 0; j < tamano_siguiente; j++) {
                    suma += deltas[capa + 1][j] * pesos[capa + 1][i][j];
                }
                deltas[capa][i] = suma * derivadaSigmoide(activaciones[capa + 1][i]);
            }
        }
        
        for(size_t capa = 0; capa < num_capas; capa++) {
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
    
    void actualizarPesos(int tamano_lote, double tasa_aprendizaje) {
        if(!hacer_backward) return;
        
        double lr = tasa_aprendizaje / tamano_lote;
        
        for(size_t capa = 0; capa < pesos.size(); capa++) {
            int tamano_entrada = pesos[capa].size();
            int tamano_salida = pesos[capa][0].size();
            
            for(int i = 0; i < tamano_entrada; i++) {
                for(int j = 0; j < tamano_salida; j++) {
                    pesos[capa][i][j] -= lr * gradientes_pesos[capa][i][j];
                    gradientes_pesos[capa][i][j] = 0.0;
                }
            }
            
            for(int j = 0; j < tamano_salida; j++) {
                bias[capa][j] -= lr * gradientes_bias[capa][j];
                gradientes_bias[capa][j] = 0.0;
            }
        }
    }
    
    double calcularError(const vector<double>& objetivo) {
        if(objetivo.size() != activaciones.back().size()) {
            return -1.0;
        }
        
        double error = 0.0;
        for(size_t i = 0; i < activaciones.back().size(); i++) {
            double diff = activaciones.back()[i] - objetivo[i];
            error += diff * diff;
        }
        
        return error * 0.5;
    }
    
    int predecir(const vector<double>& entrada) {
        vector<double> salida = propagacionHaciaAdelante(entrada);
        if(salida.empty()) return -1;
        return max_element(salida.begin(), salida.end()) - salida.begin();
    }
};

vector<vector<float>> multi_head_attention(const vector<vector<float>>& input, int num_heads, int dim_head_per_head) {
    if (input.empty() || input[0].empty()) {
        return {};
    }

    vector<vector<vector<float>>> head_outputs(num_heads);
    
    #pragma omp parallel for
    for (int h = 0; h < num_heads; h++) {
        head_outputs[h] = attention_head(input, h, dim_head_per_head);
    }
    
    int num_patches = input.size();
    int concatenated_dim = num_heads * dim_head_per_head;
    vector<vector<float>> concatenated_output(num_patches, vector<float>(concatenated_dim));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < num_patches; ++i) {
        int current_col = 0;
        for (int h = 0; h < num_heads; ++h) {
            for (int j = 0; j < dim_head_per_head; ++j) {
                if (i < head_outputs[h].size() && j < head_outputs[h][i].size()) {
                    concatenated_output[i][current_col++] = head_outputs[h][i][j];
                } else {
                    cerr << "Error: Acceso fuera de límites al concatenar salidas de heads." << endl;
                    return {};
                }
            }
        }
    }
    return concatenated_output;
}

int main() {
    cout << "Inicializando Vision Transformer..." << endl;
    
    cargar_datos();
    
    if (datos_train.empty() || datos_test.empty()) {
        cerr << "Error: No se pudieron cargar los datos" << endl;
        return 1;
    }
    
    int patch_size_dim = 49;
    int num_heads = 7;
    int dim_head_per_head = patch_size_dim / num_heads;

    if (patch_size_dim % num_heads != 0) {
        cerr << "Error: La dimensión del parche (" << patch_size_dim
             << ") no es divisible por el número de cabezas (" << num_heads << ")." << endl;
        return 1;
    }

    init_Q_K_V(num_heads, patch_size_dim, dim_head_per_head);
    
    int num_train_samples = datos_train.size();
    int num_test_samples = datos_test.size();
    
    cout << "Creando modelos MLP..." << endl;
    MLP feed_forward(patch_size_dim, patch_size_dim, 1, 128, 0);
    MLP mlp_clasificacion(patch_size_dim, 10, 2, 128, 1);

    random_device rd;
    mt19937 g(rd());

    cout << "Iniciando entrenamiento..." << endl;
    
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        cout << "--- Epoca " << epoch + 1 << " ---" << endl;
        
        vector<int> indices(num_train_samples);
        for(int i = 0; i < num_train_samples; ++i) indices[i] = i;
        shuffle(indices.begin(), indices.end(), g);

        int samples_to_process = min(static_cast<int>(indices.size()), SAMPLES_PER_EPOCH);
        int aciertos_epoch = 0;
        double error_total = 0.0;

        for (int i = 0; i < samples_to_process; i++) {
            int x = indices[i];
            
            if (x >= num_train_samples || x < 0) {
                continue;
            }
            
            vector<vector<int>> parches_originales = obtener_parches(x);
            if (parches_originales.empty()) {
                continue;
            }
            
            vector<vector<float>> parches_codificados = positional_encoding(parches_originales);
            if (parches_codificados.empty()) {
                continue;
            }

            vector<vector<float>> resultado_multi_head_attention = multi_head_attention(parches_codificados, num_heads, dim_head_per_head);
            if (resultado_multi_head_attention.empty()) {
                continue;
            }
            
            vector<vector<float>> suma_residual_attn = suma_matrices(parches_codificados, resultado_multi_head_attention);
            if (suma_residual_attn.empty()) {
                continue;
            }
            
            vector<vector<float>> normalizado_attn = layer_norm(suma_residual_attn);
            if (normalizado_attn.empty()) {
                continue;
            }

            vector<vector<float>> resultado_feed_forward_block;
            for (size_t parche_idx = 0; parche_idx < normalizado_attn.size(); parche_idx++) {
                vector<double> entrada_mlp_ff;
                for (size_t val_idx = 0; val_idx < normalizado_attn[parche_idx].size(); val_idx++) {
                    entrada_mlp_ff.push_back(static_cast<double>(normalizado_attn[parche_idx][val_idx]));
                }
                
                vector<double> salida_mlp_ff = feed_forward.propagacionHaciaAdelante(entrada_mlp_ff);
                if (salida_mlp_ff.empty()) {
                    continue;
                }
                
                vector<float> salida_float_ff;
                for (size_t val_idx = 0; val_idx < salida_mlp_ff.size(); val_idx++) {
                    salida_float_ff.push_back(static_cast<float>(salida_mlp_ff[val_idx]));
                }
                resultado_feed_forward_block.push_back(salida_float_ff);
            }
            
            if (resultado_feed_forward_block.empty()) {
                continue;
            }
            
            vector<vector<float>> suma_residual_ff = suma_matrices(normalizado_attn, resultado_feed_forward_block);
            if (suma_residual_ff.empty()) {
                continue;
            }
            
            vector<vector<float>> parches_finales = layer_norm(suma_residual_ff);
            if (parches_finales.empty()) {
                continue;
            }

            vector<double> promedio_parches(patch_size_dim, 0.0);
            for (size_t parche_idx = 0; parche_idx < parches_finales.size(); parche_idx++) {
                for (int val_idx = 0; val_idx < patch_size_dim; val_idx++) {
                    promedio_parches[val_idx] += static_cast<double>(parches_finales[parche_idx][val_idx]);
                }
            }
            for (int val_idx = 0; val_idx < patch_size_dim; val_idx++) {
                promedio_parches[val_idx] /= parches_finales.size();
            }
            
            int etiqueta_real = obtener_etiqueta(x);
            if (etiqueta_real < 0 || etiqueta_real >= 10) {
                continue;
            }
            
            vector<double> objetivo = etiqueta_a_onehot(etiqueta_real);
            
            vector<double> salida_clasificacion = mlp_clasificacion.propagacionHaciaAdelante(promedio_parches);
            if (salida_clasificacion.empty()) {
                continue;
            }
            
            mlp_clasificacion.propagacionHaciaAtras(objetivo);
            mlp_clasificacion.actualizarPesos(1, 0.01);
            
            double error = mlp_clasificacion.calcularError(objetivo);
            if (error >= 0) {
                error_total += error;
            }
            
            int prediccion = mlp_clasificacion.predecir(promedio_parches);
            if (prediccion == etiqueta_real) {
                aciertos_epoch++;
            }
            
            if (i % max(1, samples_to_process / 10) == 0) {
                cout << "Epoca " << epoch + 1 << ", Muestra " << i << "/" << samples_to_process
                     << ": Real=" << etiqueta_real << ", Pred=" << prediccion
                     << ", Error=" << fixed << setprecision(6) << error << endl;
            }
        }
        
        double accuracy_epoch = static_cast<double>(aciertos_epoch) / samples_to_process * 100.0;
        double error_promedio = error_total / samples_to_process;
        cout << "Fin Epoca " << epoch + 1 << ": Accuracy=" << fixed << setprecision(2)
             << accuracy_epoch << "%, Error promedio=" << setprecision(6) << error_promedio << endl;
    }

    cout << "\n--- Evaluando en datos de test ---" << endl;
    vector<vector<int>> matriz_confusion(10, vector<int>(10, 0));
    int aciertos_test = 0;
    double error_total_test = 0.0;

    for (int i = 0; i < num_test_samples; ++i) {
        if (i >= num_test_samples || i < 0) {
            continue;
        }

        vector<vector<int>> parches_originales = obtener_parches(i, false);
        if (parches_originales.empty()) {
            continue;
        }
        
        vector<vector<float>> parches_codificados = positional_encoding(parches_originales);
        if (parches_codificados.empty()) {
            continue;
        }

        vector<vector<float>> resultado_multi_head_attention = multi_head_attention(parches_codificados, num_heads, dim_head_per_head);
        if (resultado_multi_head_attention.empty()) {
            continue;
        }
        
        vector<vector<float>> suma_residual_attn = suma_matrices(parches_codificados, resultado_multi_head_attention);
        if (suma_residual_attn.empty()) {
            continue;
        }
        
        vector<vector<float>> normalizado_attn = layer_norm(suma_residual_attn);
        if (normalizado_attn.empty()) {
            continue;
        }

        vector<vector<float>> resultado_feed_forward_block;
        for (size_t parche_idx = 0; parche_idx < normalizado_attn.size(); parche_idx++) {
            vector<double> entrada_mlp_ff;
            for (size_t val_idx = 0; val_idx < normalizado_attn[parche_idx].size(); val_idx++) {
                entrada_mlp_ff.push_back(static_cast<double>(normalizado_attn[parche_idx][val_idx]));
            }
            
            vector<double> salida_mlp_ff = feed_forward.propagacionHaciaAdelante(entrada_mlp_ff);
            if (salida_mlp_ff.empty()) {
                continue;
            }
            
            vector<float> salida_float_ff;
            for (size_t val_idx = 0; val_idx < salida_mlp_ff.size(); val_idx++) {
                salida_float_ff.push_back(static_cast<float>(salida_mlp_ff[val_idx]));
            }
            resultado_feed_forward_block.push_back(salida_float_ff);
        }
        
        if (resultado_feed_forward_block.empty()) {
            continue;
        }
        
        vector<vector<float>> suma_residual_ff = suma_matrices(normalizado_attn, resultado_feed_forward_block);
        if (suma_residual_ff.empty()) {
            continue;
        }
        
        vector<vector<float>> parches_finales = layer_norm(suma_residual_ff);
        if (parches_finales.empty()) {
            continue;
        }

        vector<double> promedio_parches(patch_size_dim, 0.0);
        for (size_t parche_idx = 0; parche_idx < parches_finales.size(); parche_idx++) {
            for (int val_idx = 0; val_idx < patch_size_dim; val_idx++) {
                promedio_parches[val_idx] += static_cast<double>(parches_finales[parche_idx][val_idx]);
            }
        }
        for (int val_idx = 0; val_idx < patch_size_dim; val_idx++) {
            promedio_parches[val_idx] /= parches_finales.size();
        }
        
        int etiqueta_real = obtener_etiqueta(i, false);
        if (etiqueta_real < 0 || etiqueta_real >= 10) {
            continue;
        }
        
        vector<double> objetivo = etiqueta_a_onehot(etiqueta_real);
        
        int prediccion = mlp_clasificacion.predecir(promedio_parches);
        
        vector<double> salida_clasificacion = mlp_clasificacion.propagacionHaciaAdelante(promedio_parches);
        double error = mlp_clasificacion.calcularError(objetivo);
        if (error >= 0) {
            error_total_test += error;
        }

        if (prediccion == etiqueta_real) {
            aciertos_test++;
        }
        if (prediccion >= 0 && prediccion < 10) {
            matriz_confusion[etiqueta_real][prediccion]++;
        }
    }

    double accuracy_test = static_cast<double>(aciertos_test) / num_test_samples * 100.0;
    double error_promedio_test = error_total_test / num_test_samples;
    cout << "Resultado en datos de test: Accuracy=" << fixed << setprecision(2)
         << accuracy_test << "%, Error promedio=" << setprecision(6) << error_promedio_test << endl;

    cout << "\nMatriz de Confusión:" << endl;
    cout << "     ";
    for (int i = 0; i < 10; ++i) {
        cout << setw(5) << i;
    }
    cout << endl;
    cout << "-----";
    for (int i = 0; i < 10; ++i) {
        cout << "-----";
    }
    cout << endl;

    for (int i = 0; i < 10; ++i) {
        cout << setw(3) << i << " |";
        for (int j = 0; j < 10; ++j) {
            cout << setw(5) << matriz_confusion[i][j];
        }
        cout << endl;
    }

    return 0;
}