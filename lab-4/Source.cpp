#include <GL/glut.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

//parametros
constexpr int    TAM_POBLACION = 20;
constexpr int    ELITISMO = 2; // los dos mjores pasan 
constexpr double FRACCION_MUTAR = 0.30; //30% muta
constexpr int    MAX_GENERACIONES = 500; 
constexpr int    LIMITE_ESTANCAMIENTO = 100; //para detenr


//nodo para guardar posiciones selecccionadas
struct Nodo {
    float x; 
    float y;
};

//recorrido tsp
typedef std::vector<int> Recorrido;

//individuo de la pob
struct Individuo { 
    Recorrido camino; 
    double distancia; 
};

std::vector<Nodo> nodos;
std::vector<std::vector<double>> matrizDist;
std::vector<Individuo> poblacion;

Recorrido mejorRecorrido;
std::mutex mtxMejor;
bool gaEnEjecucion = false;
std::atomic<bool> necesitaRedibujar = false;


//dist euclidiana
double distancia(const Nodo& a, const Nodo& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

//crear la matriz con las distancias
void construirMatriz() {
    int n = nodos.size();
    matrizDist.assign(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            matrizDist[i][j] = matrizDist[j][i] = distancia(nodos[i], nodos[j]);
        }
    }
       
}

//suma total de distancias del recorrido
double costo(const Recorrido& recorr) {
    double d = 0;
    for (unsigned int i = 0; i + 1 < recorr.size(); ++i) {
        d += matrizDist[recorr[i]][recorr[i + 1]];
    }
    return d + matrizDist[recorr.back()][recorr.front()];
}

//seleccion por ruleta
std::vector<const Individuo*> ruleta(const std::vector<Individuo>& pob, int n, std::mt19937& rnd)
{
    //fitness
    std::vector<double>   fit(pob.size());
    std::vector<double>   esperado(pob.size());
    std::vector<double>   fraccion(pob.size());   // parte decimal
    std::vector<int>      copias(pob.size(), 0);  // copias de cada uno

    double sumaFit = 0.0;
    for (unsigned int i = 0; i < pob.size(); ++i) {
        fit[i] = 1.0 / pob[i].distancia; // 1/dist
        sumaFit += fit[i];
    }
    for (unsigned int i = 0; i < pob.size(); ++i) {
        esperado[i] = fit[i] / sumaFit * n;           // valor esperado
        copias[i] = static_cast<int>(esperado[i] + 0.5); // num copias redondeando
        fraccion[i] = esperado[i] - std::floor(esperado[i]); //parte decimal
    }

    //si faltan o sobran
    int total = 0;
    for (unsigned int i = 0; i < copias.size(); ++i) {
        total += copias[i];
    }

    //faltan
    while (total < n) {
        int mejor = -1; //indice con mejor decimal
        double maxFrac = -1.0;
        for (unsigned int i = 0; i < fraccion.size(); ++i) {
            if (fraccion[i] > maxFrac) { 
                maxFrac = fraccion[i]; 
                mejor = i;
            }
        }
        ++copias[mejor];
        fraccion[mejor] = 0.0;
        ++total;
    }

    //hay de mas
    while (total > n) {
        int peor = -1; //indice con peor decimal
        double minFrac = 2.0;
        for (unsigned int i = 0; i < fraccion.size(); ++i) {
            if (copias[i] > 0 && fraccion[i] < minFrac) {
                minFrac = fraccion[i]; 
                peor = i;
            }
        }
        --copias[peor];
        --total;
    }

    //padres
    std::vector<const Individuo*> padres;
    padres.reserve(n);
    for (unsigned int i = 0; i < pob.size(); ++i) {
        for (int c = 0; c < copias[i]; ++c) {
            padres.push_back(&pob[i]);
        }
    }
        
    while (padres.size() < static_cast<unsigned>(n))
        padres.push_back(&pob.back());
    return padres;
}

//pmx
std::pair<Recorrido, Recorrido> pmx(const Recorrido& padre1, const Recorrido& padre2, std::mt19937& rnd){
    int n = padre1.size();
    std::uniform_int_distribution<> U(0, n - 1);
    int c1 = U(rnd);
    int c2 = U(rnd);
    if (c1 > c2) std::swap(c1, c2); //invertir si c1 es mayor

    auto construirHijo = [&](const Recorrido& P1,const Recorrido& P2) -> Recorrido
    {
        Recorrido hijo(n, -1);

        //copiar el segmento de P2
        for (int i = c1; i <= c2; ++i)
            hijo[i] = P2[i];

        // mapa P2 -> P1
        std::unordered_map<int, int> mapa;
        for (int i = c1; i <= c2; ++i)
            mapa[P2[i]] = P1[i];

        // legalizar
        for (int i = 0; i < n; ++i) {
            if (i >= c1 && i <= c2) continue;

            int gen = P1[i];
            while (mapa.count(gen))
                gen = mapa[gen]; 
            hijo[i] = gen;
        }
        return hijo;
    };

    Recorrido hijo1 = construirHijo(padre1, padre2);
    Recorrido hijo2 = construirHijo(padre2, padre1);
    return { hijo1, hijo2 };
}

//mutacion orden
void mutacion(Recorrido& r, std::mt19937& rnd) {
    if (r.size() < 2) return;

    std::uniform_int_distribution<> U(0, r.size() - 1);
    int i = U(rnd); 
    int j = U(rnd);

    if (i == j) return; 
    if (i > j) {
        std::swap(i, j);
    }
    //insertar j antes de i
    int ciudad = r[j];
    r.erase(r.begin() + j);
    r.insert(r.begin() + i, ciudad);
}

//ga
void ejecutarGA()
{
    std::ofstream csv("stats.csv");
    if (!csv) { std::cerr << "no se puede abrir stats.csv\n"; return; }
    csv << "gen,mejor,promedio\n";

    //inicia poblacion inicial random
    std::mt19937 rnd(std::random_device{}());
    Recorrido base(nodos.size());
    std::iota(base.begin(), base.end(), 0);

    poblacion.clear();
    for (int i = 0; i < TAM_POBLACION; ++i) {
        std::shuffle(base.begin(), base.end(), rnd);
        poblacion.push_back({ base,costo(base) });
    }


    double mejorAnt = std::numeric_limits<double>::infinity();
    int estancado = 0;

    //loop generaciones
    for (int gen = 0; gen < MAX_GENERACIONES; ++gen) {
        //sort por distancia
        std::sort(poblacion.begin(), poblacion.end(),
            [](auto& a, auto& b) {return a.distancia < b.distancia; });


        double best = poblacion.front().distancia;
        double avg = std::accumulate(poblacion.begin(), poblacion.end(), 0.0,
            [](double s, const Individuo& ind) {return s + ind.distancia; }) / TAM_POBLACION;

        //print en consola y guardar mejor y promedio en csv
        std::cout << "Gen " << gen << " | Mejor " << best << " | Prom " << avg << "\n";
        csv << gen << ',' << best << ',' << avg << '\n'; csv.flush();

        //actualizar best si hubo mejora
        bool mejora = best < mejorAnt;
        estancado = mejora ? 0 : estancado + 1;
        if (mejora) {
            mejorAnt = best;
        } 

        //se tertmina si no se mejoro en el LIMITE
        if (estancado >= LIMITE_ESTANCAMIENTO) {
            std::cout << "Estancado " << LIMITE_ESTANCAMIENTO << " gens.\n"; break;
        }

        //mejorRecorrido para openGL
        {
            std::lock_guard<std::mutex> lk(mtxMejor);
            mejorRecorrido = poblacion.front().camino;
        }
        necesitaRedibujar = true;

        //SLEEP para visualizar
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        //elitismo
        std::vector<Individuo> nueva(poblacion.begin(),poblacion.begin() + ELITISMO);

        //seleccion
        auto padres = ruleta(poblacion,TAM_POBLACION - ELITISMO, rnd);

        //cruzamiento
        for (unsigned int k = 0; k + 1 < padres.size() && nueva.size() < TAM_POBLACION; k += 2) {
            std::pair<Recorrido, Recorrido> hijos = pmx(padres[k]->camino, padres[k + 1]->camino, rnd);
            nueva.push_back({ hijos.first,  costo(hijos.first) });
            if (nueva.size() == TAM_POBLACION) break;
            nueva.push_back({ hijos.second, costo(hijos.second) });
        }

        //mutacion
        int mutar = int(FRACCION_MUTAR * TAM_POBLACION + 0.5);
        std::uniform_int_distribution<> Uidx(ELITISMO, TAM_POBLACION - 1); //no se muta a la elite
        for (int m = 0; m < mutar; ++m) {
            int idx = Uidx(rnd);
            mutacion(nueva[idx].camino, rnd);
            nueva[idx].distancia = costo(nueva[idx].camino);
        }
        poblacion.swap(nueva); //reemplazar poblacion
    }

    csv.close();
    gaEnEjecucion = false;
    std::cout << "GA terminado.\n";
}

//openGL
void dibujar() {
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(0.2f, 0.6f, 1.0f);
    glBegin(GL_LINES);
    //aristas
    for (unsigned int i = 0; i < nodos.size(); ++i)
        for (unsigned int j = i + 1; j < nodos.size(); ++j) {
            glVertex2f(nodos[i].x, nodos[i].y);
            glVertex2f(nodos[j].x, nodos[j].y);
        }
    glEnd();

    //nodos
    glPointSize(10); glColor3f(1, 0.2f, 0.2f);
    glBegin(GL_POINTS);
    for (auto& n : nodos) glVertex2f(n.x, n.y);
    glEnd();

    //mejor recorrido
    std::lock_guard<std::mutex> lk(mtxMejor);
    if (!mejorRecorrido.empty()) {
        glLineWidth(3); glColor3f(1, 0, 0);
        glBegin(GL_LINE_LOOP);
        for (int i : mejorRecorrido) glVertex2f(nodos[i].x, nodos[i].y);
        glEnd(); glLineWidth(1);
    }
    glutSwapBuffers();
}

void redimensionar(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    gluOrtho2D(0, w, 0, h); glMatrixMode(GL_MODELVIEW);
}

void clickRaton(int b, int s, int x, int y) {
    //agregar nodo clic izq
    if (b == GLUT_LEFT_BUTTON && s == GLUT_DOWN) {
        int h = glutGet(GLUT_WINDOW_HEIGHT);
        nodos.push_back({ (float)x,(float)(h - y) });
        mejorRecorrido.clear(); glutPostRedisplay();
    }
}

void teclado(unsigned char t, int, int) {
    if (t == 'i' || t == 'I') {
        if (gaEnEjecucion) { std::cout << "inicio: .\n"; return; }
        if (nodos.size() < 3) { std::cout << "Agrega mas nodos.\n"; return; }
        construirMatriz();
        gaEnEjecucion = true;
        std::thread(ejecutarGA).detach();
    }
}

void temporizador(int) {
    if (necesitaRedibujar.exchange(false)) glutPostRedisplay();
    glutTimerFunc(16, temporizador, 0);
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(900, 700);
    glutCreateWindow("TSP con geneticos");
    glClearColor(1, 1, 1, 1);

    glutDisplayFunc(dibujar);
    glutReshapeFunc(redimensionar);
    glutMouseFunc(clickRaton);
    glutKeyboardFunc(teclado);
    glutTimerFunc(0, temporizador, 0);

    std::cout << "agregar nodos con clic izq  |   presiona i para iniciar\n";
    glutMainLoop();
    return 0;
}