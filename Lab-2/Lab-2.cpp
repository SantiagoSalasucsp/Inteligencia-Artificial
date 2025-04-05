////Windows 
#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

////MAC
//#include <OpenGL/gl.h>
//#include <OpenGL/glu.h>
//#include <GLUT/glut.h>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <queue>
#include <set>
#include <algorithm>
#include <cmath>

#include "WindowGame.h"

int main(int argc, char** argv) {
    const int boardSize = 3, winW = 600, winH = 600;
    NEnRaya game(boardSize);
    WindowGame window(game, winW, winH);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(winW, winH);
    glutCreateWindow("N en Raya");

    window.init();
    glutDisplayFunc(WindowGame::display);
    glutMouseFunc(WindowGame::mouse);
    glutMainLoop();
    return 0;
}