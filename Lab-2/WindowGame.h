#pragma once
#include "NEnRaya.h"

class WindowGame {
public:
    WindowGame(NEnRaya& g, int w, int h);

    void init() const;
    static void display();
    static void mouse(int btn, int state, int x, int y);

private:
    NEnRaya& game;
    int width, height;

    static WindowGame* instance;
};
