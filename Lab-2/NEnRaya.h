#pragma once
#include <vector>

class NEnRaya {
    int size;
    std::vector<std::vector<char>> board;

public:
    char current;

    NEnRaya(int boardSize = 3);

    void draw() const;
    void click(int x, int y, int width, int height);
};
