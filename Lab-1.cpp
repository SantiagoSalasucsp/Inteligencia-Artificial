#include <iostream>
#include <vector>
#include <cstdlib>
#include <queue>
#include <set>
#include <algorithm>
#include <cmath>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>

using namespace std;


bool compararMenor(const pair<int, double>& a, const pair<int, double>& b) {
    return a.second < b.second;
}

bool compararMayor(const pair<int, double>& a, const pair<int, double>& b) {
    return a.second > b.second;
}

class Camino {
public:
    vector<int> caminoCorrectoIds;
    vector<int> nodosVisitados;
    int memoriaMaxUsada;
    int numerosPasos;
    
    Camino() : memoriaMaxUsada(0), numerosPasos(0) {}
};

struct Arista {
    Arista(int idn = -1, int v = 0) {
        id_node = idn;
        value = v;
        pintada = false;
    }
    int id_node;
    int value;
    bool pintada;
};

template<class T>
struct Node {
    Node(int id, T v) {
        id_node = id;
        value = v;
        exists = true;
        pintado = false;
    }
    int id_node;
    T value;
    vector<Arista> edges;
    bool exists;
    bool pintado;
};

template <class T>
class CGraph {
public:
    void enlazar(Node<T>* nodo1, Node<T>* nodo2, int peso) {
        if (nodo1->exists && nodo2->exists) {
            nodo1->edges.push_back(Arista(nodo2->id_node, peso));
            nodo2->edges.push_back(Arista(nodo1->id_node, peso));
        }
    }
    
    void eliminarNodo(Node<T>* nodo) {
        if (!nodo->exists) return;
        
        for (unsigned int i = 0; i < nodo->edges.size(); i++) {
            int connected_id = nodo->edges[i].id_node;
            vector<Arista>& other_edges = nodes[connected_id].edges;
            
            for (unsigned int j = 0; j < other_edges.size(); j++) {
                if (other_edges[j].id_node == nodo->id_node) {
                    other_edges.erase(other_edges.begin() + j);
                    break;
                }
            }
        }
        
        nodo->edges.clear();
        nodo->exists = false;
    }
    
    void eliminarNodosAleatorios() {
        int total_nodes = nodes.size();
        int nodes_to_delete = total_nodes * 0.3;
        
        vector<int> indices(total_nodes);
        for (int i = 0; i < total_nodes; i++) {
            indices[i] = i;
        }
        
        for (int i = total_nodes - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            swap(indices[i], indices[j]);
        }
        
        for (int i = 0; i < nodes_to_delete; i++) {
            eliminarNodo(&nodes[indices[i]]);
        }
    }
    
    void pintarConectados(int nodoId) {
        if (!nodes[nodoId].exists) return;
        
        nodes[nodoId].pintado = true;
        
        for (unsigned int i = 0; i < nodes[nodoId].edges.size(); i++) {
            int vecinoId = nodes[nodoId].edges[i].id_node;
            if (nodes[vecinoId].exists) {
                nodes[vecinoId].pintado = true;
                nodes[nodoId].edges[i].pintada = true;
                
                for (unsigned int j = 0; j < nodes[vecinoId].edges.size(); j++) {
                    if (nodes[vecinoId].edges[j].id_node == nodoId) {
                        nodes[vecinoId].edges[j].pintada = true;
                        break;
                    }
                }
            }
        }
    }
    
    void limpiarPintado() {
        for (unsigned int i = 0; i < nodes.size(); i++) {
            nodes[i].pintado = false;
            for (unsigned int j = 0; j < nodes[i].edges.size(); j++) {
                nodes[i].edges[j].pintada = false;
            }
        }
    }
    
    CGraph() {
        for (int i = 0; i < 800; i++) {
            nodes.push_back(Node<T>(i, i));
        }
        
        int filas = 20;
        int columnas = 40;
        
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas - 1; j++) {
                int nodoActual = i * columnas + j;
                int nodoSiguiente = nodoActual + 1;
                enlazar(&nodes[nodoActual], &nodes[nodoSiguiente], 10);
            }
        }
        
        for (int j = 0; j < columnas; j++) {
            for (int i = 0; i < filas - 1; i++) {
                int nodoActual = i * columnas + j;
                int nodoSiguiente = (i + 1) * columnas + j;
                enlazar(&nodes[nodoActual], &nodes[nodoSiguiente], 10);
            }
        }
        
        for (int i = 0; i < filas - 1; i++) {
            for (int j = 0; j < columnas - 1; j++) {
                int nodoActual = i * columnas + j;
                int nodoDiagonal = (i + 1) * columnas + (j + 1);
                enlazar(&nodes[nodoActual], &nodes[nodoDiagonal],  10 * sqrt(2));
            }
        }
        
        for (int i = 0; i < filas - 1; i++) {
            for (int j = 1; j < columnas; j++) {
                int nodoActual = i * columnas + j;
                int nodoDiagonal = (i + 1) * columnas + (j - 1);
                enlazar(&nodes[nodoActual], &nodes[nodoDiagonal],  10 * sqrt(2));
            }
        }
        
        eliminarNodosAleatorios();
    }
    
    vector<Node<T>> nodes;
    
    double distanciaHeuristica(int nodo1, int nodo2) {
        int filas = 20;
        int columnas = 40;
        
        int fila1 = nodo1 / columnas;
        int col1 = nodo1 % columnas;
        
        int fila2 = nodo2 / columnas;
        int col2 = nodo2 % columnas;
        
        return sqrt(pow(fila2 - fila1, 2) + pow(col2 - col1, 2)) * 10;
    }
    
    Camino amplitud(int inicio, int fin) {
        Camino resultado;
        queue<int> cola;
        set<int> visitados;
        vector<int> padres(nodes.size(), -1);
        
        cola.push(inicio);
        visitados.insert(inicio);
        resultado.nodosVisitados.push_back(inicio);
        
        int maxMemoria = 1;
        int pasos = 0;
        bool encontrado = false;
        
        while (!cola.empty() && !encontrado) {
            if (cola.size() > maxMemoria) maxMemoria = cola.size();
            
            int actual = cola.front();
            cola.pop();
            pasos++;
            
            if (actual == fin) {
                encontrado = true;
                break;
            }
            
            for (unsigned int i = 0; i < nodes[actual].edges.size(); i++) {
                int vecino = nodes[actual].edges[i].id_node;
                
                if (visitados.find(vecino) == visitados.end()) {
                    visitados.insert(vecino);
                    resultado.nodosVisitados.push_back(vecino);
                    padres[vecino] = actual;
                    cola.push(vecino);
                    
                    if (vecino == fin) {
                        encontrado = true;
                        break;
                    }
                }
            }
        }
        
        if (encontrado) {
            int actual = fin;
            while (actual != inicio) {
                resultado.caminoCorrectoIds.push_back(actual);
                actual = padres[actual];
            }
            resultado.caminoCorrectoIds.push_back(inicio);
            reverse(resultado.caminoCorrectoIds.begin(), resultado.caminoCorrectoIds.end());
        }
        
        resultado.memoriaMaxUsada = maxMemoria;
        resultado.numerosPasos = pasos;
        return resultado;
    }
    
    Camino profundidad(int inicio, int fin) {
        Camino resultado;
        vector<int> pila;
        set<int> visitados;
        vector<int> padres(nodes.size(), -1);
        
        pila.push_back(inicio);
        visitados.insert(inicio);
        resultado.nodosVisitados.push_back(inicio);
        
        int maxMemoria = 1;
        int pasos = 0;
        bool encontrado = false;
        
        while (!pila.empty() && !encontrado) {
            if (pila.size() > maxMemoria) maxMemoria = pila.size();
            
            int actual = pila.back();
            pila.pop_back();
            pasos++;
            
            if (actual == fin) {
                encontrado = true;
                break;
            }
            
            for (int i = nodes[actual].edges.size() - 1; i >= 0; i--) {
                int vecino = nodes[actual].edges[i].id_node;
                
                if (visitados.find(vecino) == visitados.end()) {
                    visitados.insert(vecino);
                    resultado.nodosVisitados.push_back(vecino);
                    padres[vecino] = actual;
                    pila.push_back(vecino);
                    
                    if (vecino == fin) {
                        encontrado = true;
                        break;
                    }
                }
            }
        }
        
        if (encontrado) {
            int actual = fin;
            while (actual != inicio) {
                resultado.caminoCorrectoIds.push_back(actual);
                actual = padres[actual];
            }
            resultado.caminoCorrectoIds.push_back(inicio);
            reverse(resultado.caminoCorrectoIds.begin(), resultado.caminoCorrectoIds.end());
        }
        
        resultado.memoriaMaxUsada = maxMemoria;
        resultado.numerosPasos = pasos;
        return resultado;
    }
    
    Camino hillClimbing(int inicio, int fin) {
        Camino resultado;
        vector<int> listaAbierta;
        set<int> visitados;
        vector<int> padres(nodes.size(), -1);
        
        listaAbierta.push_back(inicio);
        visitados.insert(inicio);
        resultado.nodosVisitados.push_back(inicio);
        
        int maxMemoria = 1;
        int pasos = 0;
        bool encontrado = false;
        
        while (!listaAbierta.empty() && !encontrado) {
            if (listaAbierta.size() > maxMemoria) maxMemoria = listaAbierta.size();
            
            int actual = listaAbierta.front();
            listaAbierta.erase(listaAbierta.begin());
            pasos++;
            
            if (actual == fin) {
                encontrado = true;
                break;
            }
            
            vector<pair<int, double>> vecinos;
            for (unsigned int i = 0; i < nodes[actual].edges.size(); i++) {
                int vecino = nodes[actual].edges[i].id_node;
                if (visitados.find(vecino) == visitados.end()) {
                    double h = distanciaHeuristica(vecino, fin);
                    vecinos.push_back(make_pair(vecino, h));
                }
            }
            
            sort(vecinos.begin(), vecinos.end(), compararMenor);
            
            for (unsigned int i = 0; i < vecinos.size(); i++) {
                int vecino = vecinos[i].first;
                visitados.insert(vecino);
                resultado.nodosVisitados.push_back(vecino);
                padres[vecino] = actual;
                listaAbierta.insert(listaAbierta.begin(), vecino);
                
                if (vecino == fin) {
                    encontrado = true;
                    break;
                }
            }
        }
        
        if (encontrado) {
            int actual = fin;
            while (actual != inicio) {
                resultado.caminoCorrectoIds.push_back(actual);
                actual = padres[actual];
            }
            resultado.caminoCorrectoIds.push_back(inicio);
            reverse(resultado.caminoCorrectoIds.begin(), resultado.caminoCorrectoIds.end());
        }
        
        resultado.memoriaMaxUsada = maxMemoria;
        resultado.numerosPasos = pasos;
        return resultado;
    }
    
    Camino hillClimbingCorto(int inicio, int fin) {
        Camino resultado;
        vector<int> listaAbierta;
        set<int> visitados;
        vector<int> padres(nodes.size(), -1);
        
        listaAbierta.push_back(inicio);
        visitados.insert(inicio);
        resultado.nodosVisitados.push_back(inicio);
        
        int maxMemoria = 1;
        int pasos = 0;
        bool encontrado = false;
        
        while (!listaAbierta.empty() && !encontrado) {
            if (listaAbierta.size() > maxMemoria) maxMemoria = listaAbierta.size();
            
            int actual = listaAbierta.front();
            listaAbierta.erase(listaAbierta.begin());
            pasos++;
            
            if (actual == fin) {
                encontrado = true;
                break;
            }
            
            vector<pair<int, double>> vecinos;
            for (unsigned int i = 0; i < nodes[actual].edges.size(); i++) {
                int vecino = nodes[actual].edges[i].id_node;
                if (visitados.find(vecino) == visitados.end()) {
                    double h = distanciaHeuristica(vecino, fin);
                    vecinos.push_back(make_pair(vecino, h));
                }
            }
            
            sort(vecinos.begin(), vecinos.end(), compararMayor);
            
            for (unsigned int i = 0; i < vecinos.size(); i++) {
                int vecino = vecinos[i].first;
                visitados.insert(vecino);
                resultado.nodosVisitados.push_back(vecino);
                padres[vecino] = actual;
                listaAbierta.insert(listaAbierta.begin(), vecino);
            }
        }
        
        if (encontrado) {
            int actual = fin;
            while (actual != inicio) {
                resultado.caminoCorrectoIds.push_back(actual);
                actual = padres[actual];
            }
            resultado.caminoCorrectoIds.push_back(inicio);
            reverse(resultado.caminoCorrectoIds.begin(), resultado.caminoCorrectoIds.end());
        }
        
        resultado.memoriaMaxUsada = maxMemoria;
        resultado.numerosPasos = pasos;
        return resultado;
    }
    
    void sumarValores(double& x, double y) {
        x += y;
    }

    Camino aestrella(int inicio, int fin) {
        Camino resultado;
        vector<int> cola;
        set<int> visitados;
        vector<int> padres(nodes.size(), -1);
        vector<double> costos(nodes.size(), 20);
        
        cola.push_back(inicio);
        visitados.insert(inicio);
        resultado.nodosVisitados.push_back(inicio);
        costos[inicio] = 0;
        
        int maxMemoria = 1;
        int pasos = 0;
        bool encontrado = false;
        
        while (!cola.empty() && !encontrado) {
            if (cola.size() > maxMemoria)
                maxMemoria = cola.size();
            
            int mejorIdx = 0;
            double mejorValor = costos[cola[0]] + distanciaHeuristica(cola[0], fin);
            
            for (size_t i = 1; i < cola.size(); ++i) {
                double valorActual = costos[cola[i]] + distanciaHeuristica(cola[i], fin);
                if (valorActual < mejorValor) {
                    mejorValor = valorActual;
                    mejorIdx = i;
                }
            }
            
            int actual = cola[mejorIdx];
            cola.erase(cola.begin() + mejorIdx);
            pasos++;
            
            if (actual == fin) {
                encontrado = true;
                break;
            }
            
            for (size_t i = 0; i < nodes[actual].edges.size(); ++i) {
                int vecino = nodes[actual].edges[i].id_node;
                double nuevoCosto = costos[actual] + nodes[actual].edges[i].value;
                
                if (visitados.find(vecino) == visitados.end() || nuevoCosto < costos[vecino]) {
                    double costoTotal = nuevoCosto;
                    sumarValores(costoTotal, distanciaHeuristica(vecino, fin));
                    
                    visitados.insert(vecino);
                    resultado.nodosVisitados.push_back(vecino);
                    padres[vecino] = actual;
                    costos[vecino] = nuevoCosto;
                    cola.push_back(vecino);
                    
                    if (vecino == fin) {
                        encontrado = true;
                        break;
                    }
                }
            }
        }
        
        if (encontrado) {
            int actual = fin;
            while (actual != inicio) {
                resultado.caminoCorrectoIds.push_back(actual);
                actual = padres[actual];
            }
            resultado.caminoCorrectoIds.push_back(inicio);
            reverse(resultado.caminoCorrectoIds.begin(), resultado.caminoCorrectoIds.end());
        }
        
        resultado.memoriaMaxUsada = maxMemoria;
        resultado.numerosPasos = pasos;
        return resultado;
    }
};

CGraph<int> g;
int primerNodoSeleccionado = -1;
int segundoNodoSeleccionado = -1;
int anchoPantalla = 800;
int altoPantalla = 400;
int columnas = 40;

int encontrarNodoCercano(int x, int y) {
    float gx = (float)x / anchoPantalla * 400;
    float gy = (float)(altoPantalla - y) / altoPantalla * 200;

    int nodoMasCercano = -1;
    float distanciaMinima = 10000.0f;

    for (int i = 0; i < g.nodes.size(); i++) {
        if (!g.nodes[i].exists) continue;

        float nx = (g.nodes[i].id_node % columnas) * 10;
        float ny = (g.nodes[i].id_node / columnas) * 10;

        float dist = sqrt((nx-gx)*(nx-gx) + (ny-gy)*(ny-gy));

        if (dist < distanciaMinima && dist < 8) {
            distanciaMinima = dist;
            nodoMasCercano = i;
        }
    }

    return nodoMasCercano;
}

void manejarClic(int boton, int estado, int x, int y) {
    if (boton == GLUT_LEFT_BUTTON && estado == GLUT_DOWN) {
        int nodoId = encontrarNodoCercano(x, y);

        if (nodoId != -1) {
            if (primerNodoSeleccionado == -1) {
                primerNodoSeleccionado = nodoId;
                cout << "Nodo inicio seleccionado: " << nodoId << endl;
            }
            else if (segundoNodoSeleccionado == -1 && nodoId != primerNodoSeleccionado) {
                segundoNodoSeleccionado = nodoId;
                cout << "Nodo fin seleccionado: " << nodoId << endl;

                
                //Camino resultado = g.amplitud(primerNodoSeleccionado, segundoNodoSeleccionado);
                //Camino resultado = g.profundidad(primerNodoSeleccionado, segundoNodoSeleccionado);
                //Camino resultado = g.hillClimbing(primerNodoSeleccionado, segundoNodoSeleccionado);
                //Camino resultado = g.hillClimbingCorto(primerNodoSeleccionado, segundoNodoSeleccionado);
                Camino resultado = g.aestrella(primerNodoSeleccionado, segundoNodoSeleccionado);

                cout << "Pasos: " << resultado.numerosPasos << ", Memoria maxima: " << resultado.memoriaMaxUsada << endl;

                for (unsigned int i = 0; i < resultado.nodosVisitados.size(); i++) {
                    g.nodes[resultado.nodosVisitados[i]].pintado = true;
                }

                for (unsigned int i = 0; i < resultado.caminoCorrectoIds.size(); i++) {
                    int id = resultado.caminoCorrectoIds[i];
                    g.nodes[id].pintado = true;
                    
                    for (unsigned int j = 0; j < g.nodes[id].edges.size(); j++) {
                        int vecino = g.nodes[id].edges[j].id_node;
                        if (find(resultado.caminoCorrectoIds.begin(), resultado.caminoCorrectoIds.end(), vecino) != resultado.caminoCorrectoIds.end()) {
                            g.nodes[id].edges[j].pintada = true;
                        }
                    }
                }
            }
            else {
                g.limpiarPintado();
                primerNodoSeleccionado = nodoId;
                segundoNodoSeleccionado = -1;
                cout << "Nueva selección. Nodo inicio seleccionado: " << nodoId << endl;
            }

            glutPostRedisplay();
        }
    }
}

void dibujar() {
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 400, 0, 200);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glLineWidth(1.0);
    glBegin(GL_LINES);
    for (unsigned int i = 0; i < g.nodes.size(); i++) {
        if (!g.nodes[i].exists) continue;

        float x1 = (g.nodes[i].id_node % columnas) * 10;
        float y1 = (g.nodes[i].id_node / columnas) * 10;

        for (unsigned int j = 0; j < g.nodes[i].edges.size(); j++) {
            int vecinoId = g.nodes[i].edges[j].id_node;
            if (!g.nodes[vecinoId].exists || vecinoId <= g.nodes[i].id_node) continue;

            float x2 = (vecinoId % columnas) * 10;
            float y2 = (vecinoId / columnas) * 10;

            if (g.nodes[i].edges[j].pintada) {
                glColor3f(0.0, 1.0, 0.0);
            } else {
                glColor3f(0.0, 0.0, 1.0);
            }

            glVertex2f(x1, y1);
            glVertex2f(x2, y2);
        }
    }
    glEnd();

    glPointSize(6.0);
    glBegin(GL_POINTS);
    for (unsigned int i = 0; i < g.nodes.size(); i++) {
        if (!g.nodes[i].exists) continue;

        if (i == primerNodoSeleccionado) {
            glColor3f(0.0, 1.0, 0.0);
        } else if (i == segundoNodoSeleccionado || g.nodes[i].pintado) {
            glColor3f(1.0, 0.0, 0.0);
        } else {
            glColor3f(0.0, 1.0, 1.0);
        }

        float x = (g.nodes[i].id_node % columnas) * 10;
        float y = (g.nodes[i].id_node / columnas) * 10;
        glVertex2f(x, y);
    }
    glEnd();

    glutSwapBuffers();
}

void inicializar() {
    glClearColor(0.0, 0.0, 0.0, 0.0);
}

int main(int argc, char** argv) {
    srand(time(NULL));

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(anchoPantalla, altoPantalla);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Visualizacion de Grafo");

    inicializar();
    glutDisplayFunc(dibujar);
    glutMouseFunc(manejarClic);

    g = CGraph<int>();

    glutMainLoop();
    return 0;
}


