#include "WindowGame.h"

//WINDOWS
#include <windows.h>
#include <GL/glut.h>


WindowGame* WindowGame::instance = nullptr;

WindowGame::WindowGame(NEnRaya& g, int w, int h, bool iaFirst, int depth)
    : game(g), width(w), height(h), iaTurn(iaFirst), maxDepth(depth) {
    instance = this;
    reset();
}

void WindowGame::init() const {
    glClearColor(0, 0, 0, 1);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0, 1, 0, 1);
}

void WindowGame::display() {
    if (instance) {
        instance->game.draw();
        if (instance->iaTurn) instance->handleIATurn();
    }
}

void WindowGame::mouse(int b, int s, int x, int y) {
    if (instance && b == GLUT_LEFT_BUTTON && s == GLUT_DOWN && !instance->iaTurn) {
        int cellW = instance->width / instance->game.getSize();
        int cellH = instance->height / instance->game.getSize();
        int col = x / cellW, row = (instance->height - y) / cellH;
        if (row >= 0 && row < instance->game.getSize() && col >= 0 && col < instance->game.getSize() &&
            instance->game.getBoard()[row][col] == ' ') {

            instance->game.click(x, y, instance->width, instance->height);
            delete instance->tree;
            instance->tree = new GameTreeNode(instance->game.getBoard(), 'X', instance->maxDepth);
            instance->tree->generateChildren();
            instance->iaTurn = true;
            glutPostRedisplay();
        }
    }
}

void WindowGame::handleIATurn() {
    GameTreeNode* best = tree->bestMove(true);
    if (best && best != tree) {
        game.setBoard(best->board);
        iaTurn = false;
        delete tree;
        tree = new GameTreeNode(game.getBoard(), 'X', maxDepth);
        tree->generateChildren();
        glutPostRedisplay();
    }
}

void WindowGame::reset() {
    game.reset();
    if (tree) delete tree;
    tree = new GameTreeNode(game.getBoard(), 'X', maxDepth);
    tree->generateChildren();
}