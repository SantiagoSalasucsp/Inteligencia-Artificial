#pragma once
#include <vector>

struct GameTreeNode {
    std::vector<std::vector<char>> board;
    char currentPlayer;
    int aiWinPaths, playerWinPaths;
    int depth;
    std::vector<GameTreeNode*> children;

    GameTreeNode(const std::vector<std::vector<char>>& b, char player, int d);
    ~GameTreeNode();

    void generateChildren();
    void evaluateWinPaths();
    int utility() const;
    GameTreeNode* bestMove(bool maximizing);
};