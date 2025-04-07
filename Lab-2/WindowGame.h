#pragma once
#include "NEnRaya.h"
#include "GameTreeNode.h"

class WindowGame {
public:
    WindowGame(NEnRaya* g, int w, int h, bool iaFirst, int depth);
    ~WindowGame();
    void init() const;

    //Dibujar y jugar el N en raya
    static void display();
    static void mouse(int btn, int state, int x, int y);

    //Configurar el N en Raya
    static void configGame();
    static void configMouse(int button, int state, int x, int y);

private:
    static WindowGame* instance;
    NEnRaya* game;
    GameTreeNode* tree;
    int width, height;
    bool iaTurn;
    int maxDepth;

    void handleIATurn();
    void reset(int s);

    // Variables para la configuración
    int configBoardSize;
    int configDepth;
    bool configIaFirst;
    void drawString(float x, float y, const char* string) const;
};
