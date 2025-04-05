#include "NEnRaya.h"
#include <cmath>
#include <algorithm>

//WINDOWS
#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

NEnRaya::NEnRaya(int boardSize) : size(boardSize), board(size, std::vector<char>(size, ' ')), current('O') {}

const std::vector<std::vector<char>>& NEnRaya::getBoard() const {
    return board;
}

void NEnRaya::setBoard(const std::vector<std::vector<char>>& b) {
    board = b;
    current = 'O';
}

void NEnRaya::reset() {
    for (auto& row : board)
        std::fill(row.begin(), row.end(), ' ');
    current = 'O';
}

int NEnRaya::getCell(int row, int col) const {
    return board[row][col];
}

int NEnRaya::getSize() const {
    return size;
}

void NEnRaya::click(int x, int y, int width, int height) {
    int cellW = width / size, cellH = height / size;
    int col = x / cellW, row = (height - y) / cellH;
    if (row >= 0 && row < size && col >= 0 && col < size && board[row][col] == ' ') {
        board[row][col] = current;
        current = (current == 'X') ? 'O' : 'X';
    }
}

void NEnRaya::draw() const {
    const float step = 1.0f / size, pad = step * 0.2f, radius = step * 0.3f;
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1, 1, 1);
    glLineWidth(2);
    glBegin(GL_LINES);
    for (int i = 1; i < size; ++i) {
        float pos = i * step;
        glVertex2f(pos, 0); glVertex2f(pos, 1);
        glVertex2f(0, pos); glVertex2f(1, pos);
    }
    glEnd();

    for (int r = 0; r < size; ++r)
        for (int c = 0; c < size; ++c) {
            float x = c * step, y = r * step;
            switch (board[r][c]) {
            case 'X':
                glColor3f(1, 0, 0);
                glBegin(GL_LINES);
                glVertex2f(x + pad, y + pad);
                glVertex2f(x + step - pad, y + step - pad);
                glVertex2f(x + pad, y + step - pad);
                glVertex2f(x + step - pad, y + pad);
                glEnd();
                break;
            case 'O':
                glColor3f(0, 0, 1);
                glBegin(GL_LINE_LOOP);
                for (int i = 0; i < 100; ++i) {
                    float angle = 2 * 3.14159f * i / 100;
                    glVertex2f(x + step / 2 + radius * cosf(angle),
                        y + step / 2 + radius * sinf(angle));
                }
                glEnd();
                break;
            }
        }
    glutSwapBuffers();
}
