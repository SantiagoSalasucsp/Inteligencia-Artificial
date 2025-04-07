#pragma once
#include <vector>

class NEnRaya {
    int size;
    std::vector<std::vector<char>> board;

public:
    char current;

    NEnRaya(int boardSize = 3);
    const std::vector<std::vector<char>>& getBoard() const;
    void setBoard(const std::vector<std::vector<char>>& b);
    void reset(int s);
    void click(int x, int y, int w, int h);
    void draw() const;
    int getCell(int row, int col) const;
    int getSize() const;
};
