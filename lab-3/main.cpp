#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <set>
#include <algorithm>
#include <numeric>

#define M_PI 3.14159265
#define MAX_COLOR 3

using namespace std;

struct Nodo {
    float x = 0.0f, y = 0.0f;
    float radio = 0.07f;
    int color = 1;
    vector<int> vecinos;
};

vector<Nodo> grafo;
int selectedCircle = -1;
vector<int> orden;

bool isColorValid(int nodeIndex, int color) {
    for (int neighbor : grafo[nodeIndex].vecinos) {
        if (grafo[neighbor].color == color) return false;
    }
    return true;
}

bool backtrack(int index) {
    if (index == orden.size()) return true;

    int currentNode = orden[index];
    for (int color = 1; color <= MAX_COLOR; ++color) {
        if (isColorValid(currentNode, color)) {
            grafo[currentNode].color = color;
            if (backtrack(index + 1)) return true;
            grafo[currentNode].color = 0;
        }
    }

    return false;
}

void Restrictivo() {
    for (auto& n : grafo) n.color = 0;

    orden.resize(grafo.size());
    for (int i = 0; i < orden.size(); i++) { orden[i] = i; }

    sort(orden.begin(), orden.end(), [](int a, int b) {
        return grafo[a].vecinos.size() > grafo[b].vecinos.size();
        });

    backtrack(0);
}

void Restringida() {
    set<int> visitados;
    orden.clear();

    for (int i = 0; i < grafo.size(); ++i) {
        if (visitados.count(i)) continue;

        vector<int> cola = { i };
        visitados.insert(i);

        while (!cola.empty()) {
            int actual = cola.front();
            cola.erase(cola.begin());
            orden.push_back(actual);

            for (int vecino : grafo[actual].vecinos) {
                if (!visitados.count(vecino)) {
                    visitados.insert(vecino);
                    cola.push_back(vecino);
                }
            }
        }
    }

    for (int i : orden) {
        grafo[i].color = 0;
    }

    backtrack(0);
}

void drawCircle(float x, float y, float radius, int color) {
    const int segments = 50;
    switch (color) {
    case 1: glColor3f(0.0f, 1.0f, 0.0f); break;
    case 2: glColor3f(1.0f, 0.0f, 0.0f); break;
    case 3: glColor3f(0.0f, 0.0f, 1.0f); break;
    default: glColor3f(0.5f, 0.5f, 0.5f); break;
    }

    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(x, y);
    for (int i = 0; i <= segments; ++i) {
        float angle = 2.0f * M_PI * i / segments;
        float dx = cos(angle) * radius;
        float dy = sin(angle) * radius;
        glVertex2f(x + dx, y + dy);
    }
    glEnd();
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        double xpos, ypos;
        int width, height;
        glfwGetCursorPos(window, &xpos, &ypos);
        glfwGetWindowSize(window, &width, &height);

        float glX = static_cast<float>((xpos / width) * 2.0 - 1.0);
        float glY = static_cast<float>(-((ypos / height) * 2.0 - 1.0));

        for (int i = 0; i < grafo.size(); ++i) {
            float dx = glX - grafo[i].x;
            float dy = glY - grafo[i].y;
            if (std::sqrt(dx * dx + dy * dy) <= grafo[i].radio) {
                if (selectedCircle != -1 && selectedCircle != i) {
                    if (std::find(grafo[selectedCircle].vecinos.begin(), grafo[selectedCircle].vecinos.end(), i) == grafo[selectedCircle].vecinos.end()) {
                        grafo[selectedCircle].vecinos.push_back(i);
                        grafo[i].vecinos.push_back(selectedCircle);
                        Restrictivo();
                        //Restringida();
                    }
                    selectedCircle = -1;
                }
                else {
                    selectedCircle = i;
                }
                return;
            }
        }

        Nodo nuevo;
        nuevo.x = glX;
        nuevo.y = glY;
        nuevo.color = 1;
        grafo.push_back(nuevo);

        if (!grafo.back().vecinos.empty()) {
            Restrictivo();
            //Restringida();
        }
    }
}

int main() {
    if (!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(800, 600, "Grafo con Coloración", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glewInit();

    glfwSetMouseButtonCallback(window, mouse_button_callback);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        glColor3f(0.2f, 0.2f, 0.2f);
        glBegin(GL_LINES);
        for (int i = 0; i < grafo.size(); ++i) {
            for (int j : grafo[i].vecinos) {
                if (i < j) {
                    glVertex2f(grafo[i].x, grafo[i].y);
                    glVertex2f(grafo[j].x, grafo[j].y);
                }
            }
        }
        glEnd();

        for (const auto& n : grafo) {
            drawCircle(n.x, n.y, n.radio, n.color);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}