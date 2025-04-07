#include "GameTreeNode.h"
#include <algorithm>
#include <cmath>
#include <iostream>

GameTreeNode::GameTreeNode(const std::vector<std::vector<char>>& b, char player, int d)
    : board(b), currentPlayer(player), depth(d), aiWinPaths(0), playerWinPaths(0) {
    evaluateWinPaths();
}

GameTreeNode::~GameTreeNode() {
    for (auto* child : children) delete child;
}

//Crea las jugadas hijas en el arbol
void GameTreeNode::generateChildren() {
    if (depth <= 0) return;
    int size = board.size();
    for (int r = 0; r < size; ++r)
        for (int c = 0; c < size; ++c)
            if (board[r][c] == ' ') {
                auto next = board;
                next[r][c] = currentPlayer;
                children.push_back(new GameTreeNode(next, currentPlayer == 'X' ? 'O' : 'X', depth - 1));
            }
    for (auto* child : children) child->generateChildren();
}

//Evalua las formas deganar de IA - Las formas de ganar jugador
void GameTreeNode::evaluateWinPaths() {
    int size = board.size();
    auto count = [&](char p) {
        int total = 0;
        for (int i = 0; i < size; ++i) {
            if (std::all_of(board[i].begin(), board[i].end(), [&](char c) { return c == ' ' || c == p; })) ++total;
            bool col = true;
            for (int j = 0; j < size; ++j)
                if (board[j][i] != ' ' && board[j][i] != p) col = false;
            if (col) ++total;
        }
        bool diag1 = true, diag2 = true;
        for (int i = 0; i < size; ++i) {
            if (board[i][i] != ' ' && board[i][i] != p) diag1 = false;
            if (board[i][size - 1 - i] != ' ' && board[i][size - 1 - i] != p) diag2 = false;
        }
        if (diag1) ++total;
        if (diag2) ++total;
        return total;
        };
    aiWinPaths = count('X');
    playerWinPaths = count('O');
}

int GameTreeNode::utility() const {
    return aiWinPaths - playerWinPaths;
}

//Busca la mejor jugada segun minmax en todo el arbol generado con profundidad "depth"
GameTreeNode* GameTreeNode::bestMove(bool maximizing) {
    if (children.empty()) return this;
    int bestVal = maximizing ? -9999 : 9999;
    GameTreeNode* best = nullptr;
    for (auto* child : children) {
        //imprimir info tabla
        for (int i = 0; i < child->depth; i++) std::cout << " ";
        std::cout << "D " << child->depth << ": " << child->aiWinPaths << " - " << child->playerWinPaths << " = " << child->utility() << " " << &*child  << "\n";
        //imprimir tabla
        for (int r = 0; r < child->board.size(); r++) {
            for (int c = 0; c < child->board.size(); c++)
                std::cout << (child->board[r][c]);
            std::cout << "\n";
        }
        //buscar minmax
        int val = child->bestMove(!maximizing)->utility();
        if ((maximizing && val > bestVal) || (!maximizing && val < bestVal)) {
            bestVal = val;
            best = child;
            //imprimir best
            std::cout << "======\nBEST: " << &*child <<" "  << maximizing << "\n";
            //imprimir info tabla
            for (int i = 0; i < child->depth; i++) std::cout << " ";
            std::cout << "D " << child->depth << ": " << child->aiWinPaths << " - " << child->playerWinPaths << " = " << child->utility() << " " << &*child << "\n";
            //imprimir tabla
            for (int r = 0; r < child->board.size(); r++) {
                for (int c = 0; c < child->board.size(); c++)
                    std::cout << (child->board[r][c]);
                std::cout << "\n";
            }
            std::cout << "======\n\n";
        }
    }
    return best;
}