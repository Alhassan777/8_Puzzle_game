# 8-Puzzle Game Project
![Uploading Screenshot 2025-03-13 at 12.59.04 AM.png…]()

## Overview

This project implements an interactive 8-puzzle game using Python and Pygame. The game allows users to shuffle the puzzle, manually solve it, or use various algorithms to find a solution automatically. The project consists of two main files: `puzzle_solver.py` and `puzzle_game.py`.

## puzzle_solver.py

### Purpose

The `puzzle_solver.py` file contains the logic for solving the 8-puzzle using different algorithms. It includes implementations for Breadth-First Search (BFS), Depth-First Search (DFS), Uniform-Cost Search (UCS), and A* Search with both Graph and Tree search strategies. Additionally, it provides three heuristic functions to guide the A* search.

### Key Components

1. **Heuristics**:
   - `h1`: Counts the number of misplaced tiles.
   - `h2`: Calculates the sum of Manhattan distances for each tile.
   - `h3`: Combines Manhattan distance with linear conflict detection.

2. **PuzzleNode Class**:
   - Represents a state in the puzzle with attributes for tracking parent nodes, costs, and generating child nodes.

3. **Solvability Checker**:
   - Determines if a given puzzle configuration is solvable based on inversion counts and blank tile position.

4. **Search Algorithms**:
   - `solvePuzzleBFS`: Implements BFS to find a solution path.
   - `solvePuzzleDFS`: Implements DFS for completeness.
   - `solvePuzzleUCS`: Uses UCS to expand nodes based on cost.
   - `solvePuzzleAStarGraph`: A* Graph Search with selectable heuristics.
   - `solvePuzzleAStarTree`: A* Tree Search without tracking visited states.

## puzzle_game.py

### Purpose

The `puzzle_game.py` file provides the graphical interface for the 8-puzzle game using Pygame. It manages the game state, user interactions, and visual representation of the puzzle and buttons.

### Key Components

1. **PuzzleGame Class**:
   - Manages the puzzle state, shuffling, solving, and animation.
   - Provides methods to shuffle the puzzle, move tiles, and solve using selected algorithms.

2. **Drawing Functions**:
   - `draw_tile`: Renders individual tiles with colors and shadows.
   - `draw_button`: Creates interactive buttons with hover effects.
   - `draw_info`: Displays game information and solution statistics.

3. **Main Loop**:
   - Initializes the game, handles user input, and updates the display.
   - Provides buttons for shuffling, reverting, changing speed, and selecting algorithms.

## How to Run

1. Ensure Python and Pygame are installed on your system.
2. Run the `puzzle_game.py` file to start the game interface.
3. Use the buttons to shuffle the puzzle, solve it manually, or select an algorithm for automatic solving.

## Conclusion

This project demonstrates the integration of algorithmic problem-solving with graphical user interfaces. The `puzzle_solver.py` file provides robust solutions for the 8-puzzle, while `puzzle_game.py` offers an engaging way to interact with these solutions visually.
