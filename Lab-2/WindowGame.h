#pragma once
#include "NEnRaya.h"
#include "GameTreeNode.h"

class WindowGame {
public:
    WindowGame(NEnRaya& g, int w, int h, bool iaFirst, int depth);
    void init() const;
    static void display();
    static void mouse(int btn, int state, int x, int y);

private:
    static WindowGame* instance;
    NEnRaya& game;
    int width, height;
    GameTreeNode* tree;
    bool iaTurn;
    int maxDepth;

    void handleIATurn();
    void reset();
};
