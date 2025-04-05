#include "GameTreeNode.h"
#include <algorithm>
#include <cmath>

GameTreeNode::GameTreeNode(const std::vector<std::vector<char>>& b, char player, int d)
    : board(b), currentPlayer(player), depth(d), aiWinPaths(0), playerWinPaths(0) {
    evaluateWinPaths();
}

GameTreeNode::~GameTreeNode() {
    for (auto* child : children) delete child;
}

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

GameTreeNode* GameTreeNode::bestMove(bool maximizing) {
    if (children.empty()) return this;
    int bestVal = maximizing ? -9999 : 9999;
    GameTreeNode* best = nullptr;
    for (auto* child : children) {
        int val = child->bestMove(!maximizing)->utility();
        if ((maximizing && val > bestVal) || (!maximizing && val < bestVal)) {
            bestVal = val;
            best = child;
        }
    }
    return best;
}