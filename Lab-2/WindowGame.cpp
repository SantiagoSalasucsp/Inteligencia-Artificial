#include "WindowGame.h"
#include <windows.h>
#include <GL/glut.h>

WindowGame* WindowGame::instance = nullptr;

WindowGame::WindowGame(NEnRaya& g, int w, int h) : game(g), width(w), height(h) {
    instance = this;
}

void WindowGame::init() const {
    glClearColor(0, 0, 0, 1);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0, 1, 0, 1);
}

void WindowGame::display() {
    if (instance) instance->game.draw();
}

void WindowGame::mouse(int b, int s, int x, int y) {
    if (instance && b == GLUT_LEFT_BUTTON && s == GLUT_DOWN) {
        instance->game.click(x, y, instance->width, instance->height);
        glutPostRedisplay();
    }
}