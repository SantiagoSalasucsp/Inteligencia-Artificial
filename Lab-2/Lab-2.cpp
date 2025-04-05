#include "WindowGame.h"

//WINDOWS
#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

////MAC
//#include <OpenGL/gl.h>
//#include <OpenGL/glu.h>
//#include <GLUT/glut.h>

int main(int argc, char** argv) {
    const int boardSize = 3, width = 600, height = 600, depth = 1;
    bool iaFirst = true;

    NEnRaya game(boardSize);
    WindowGame window(game, width, height, iaFirst, depth);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(width, height);
    glutCreateWindow("IA vs Jugador - N en Raya");

    window.init();
    glutDisplayFunc(WindowGame::display);
    glutMouseFunc(WindowGame::mouse);
    glutMainLoop();
    return 0;
}
