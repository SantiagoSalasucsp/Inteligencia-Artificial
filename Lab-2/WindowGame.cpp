#include "WindowGame.h"

//WINDOWS
#include <windows.h>
#include <GL/glut.h>
#include <cstdio> // para sprintf

#define sprintf sprintf_s

WindowGame* WindowGame::instance = nullptr;

// inicializa el parametros
WindowGame::WindowGame(NEnRaya* g, int w, int h, bool iaFirst, int depth)
    : game(g), width(w), height(h), iaTurn(iaFirst), maxDepth(depth), 
    configBoardSize(g->getSize()), configDepth(depth), configIaFirst(iaFirst) {
    instance = this;
    reset(g->getSize());
}

WindowGame::~WindowGame() {
    if(game) delete game;
    if(tree) delete tree;
    delete instance;
}

void WindowGame::init() const {
    glClearColor(0, 0, 0, 1);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0, 1, 0, 1);
}

//Dibuja el tablero y juega la IA
void WindowGame::display() {
    if (instance) {
        instance->game->draw();
        if (instance->iaTurn) instance->handleIATurn();
    }
}

// Selecciona tu jugada en la tabla
void WindowGame::mouse(int b, int s, int x, int y) {
    if (instance && b == GLUT_LEFT_BUTTON && s == GLUT_DOWN && !instance->iaTurn) {
        int cellW = instance->width / instance->game->getSize();
        int cellH = instance->height / instance->game->getSize();
        int col = x / cellW, row = y / cellH;
        if (row >= 0 && row < instance->game->getSize() && col >= 0 && col < instance->game->getSize() &&
            instance->game->getBoard()[row][col] == ' ') {

            instance->game->click(x, y, instance->width, instance->height);
            delete instance->tree;
            instance->tree = new GameTreeNode(instance->game->getBoard(), 'X', instance->maxDepth);
            instance->tree->generateChildren();
            instance->iaTurn = true;
            glutPostRedisplay();
        }
    }
}

//Juega la IA
void WindowGame::handleIATurn() {
    GameTreeNode* best = tree->bestMove(true);
    if (best && best != tree) {
        game->setBoard(best->board);
        iaTurn = false;
        delete tree;
        tree = new GameTreeNode(game->getBoard(), 'O', maxDepth);
        tree->generateChildren();
        glutPostRedisplay();
    }
}

//reseta el juego
void WindowGame::reset(int s) {
    game->reset(configBoardSize);
    if (tree) delete tree;
    tree = new GameTreeNode(game->getBoard(), 'X', maxDepth);
    tree->generateChildren();
}


//dibujar texto
void WindowGame::drawString(float x, float y, const char* string) const {
    glRasterPos2f(x, y);
    while (*string) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *string);
        ++string;
    }
}

// Pantalla de configuracion
void WindowGame::configGame(){
    glClear(GL_COLOR_BUFFER_BIT);
    char buf[100];

    // opcion Board Size
    sprintf(buf, "Board Size: %d (RMB: +, LMB: -)", instance->configBoardSize);
    glColor3f(1, 1, 1);
    instance->drawString(0.1f, 0.8f, buf);

    // opcion Depth
    sprintf(buf, "Depth: %d (RMB: +, LMB: -)", instance->configDepth);
    instance->drawString(0.1f, 0.7f, buf);

    // opcion IA First
    sprintf(buf, "IA First: %s (Click to toggle)", instance->configIaFirst ? "true" : "false");
    instance->drawString(0.1f, 0.6f, buf);

    // Dibujar boton "Start"
    glColor3f(0.5f, 0.5f, 0.5f);
    glBegin(GL_QUADS);
    glVertex2f(0.8f, 0.1f);
    glVertex2f(0.95f, 0.1f);
    glVertex2f(0.95f, 0.2f);
    glVertex2f(0.8f, 0.2f);
    glEnd();
    glColor3f(1, 1, 1);
    instance->drawString(0.82f, 0.15f, "Start");

    glutSwapBuffers();
}

// Callback del mouse en configuracion
void WindowGame::configMouse(int button, int state, int x, int y) {
    if (state != GLUT_DOWN) return;
    float normX = float(x) / instance->width;
    float normY = 1.0f - float(y) / instance->height;

    // area Board Size 
    if (normX >= 0.1f && normX <= 0.6f && normY >= 0.8f && normY <= 0.85f) {
        if (button == GLUT_LEFT_BUTTON && instance->configBoardSize > 1)
            instance->configBoardSize--;
        else if (button == GLUT_RIGHT_BUTTON)
            instance->configBoardSize++;
        glutPostRedisplay();
        return;
    }

    // area para Depth
    if (normX >= 0.1f && normX <= 0.6f && normY >= 0.7f && normY <= 0.75f) {
        if (button == GLUT_LEFT_BUTTON && instance->configDepth > 1)
            instance->configDepth--;
        else if (button == GLUT_RIGHT_BUTTON)
            instance->configDepth++;
        glutPostRedisplay();
        return;
    }

    // area para IAFirst 
    if (normX >= 0.1f && normX <= 0.6f && normY >= 0.6f && normY <= 0.65f) {
        instance->configIaFirst = !instance->configIaFirst;
        glutPostRedisplay();
        return;
    }

    // area para boton Start 
    if (normX >= 0.8f && normX <= 0.95f && normY >= 0.1f && normY <= 0.2f) {
        instance->maxDepth = instance->configDepth;
        instance->iaTurn = instance->configIaFirst;
        instance->reset(instance->configBoardSize);
        // Cambiar los callbacks para cambiar del modo configuracion al juego
        glutMouseFunc(WindowGame::mouse);
        glutDisplayFunc(WindowGame::display);
        glutPostRedisplay();
        return;
    }
}
