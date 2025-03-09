# 8-Puzzle Game

This repository contains an implementation of the classic 8-puzzle game with a graphical user interface built using Pygame. The application not only allows users to play the game but also demonstrates various search algorithms used in artificial intelligence to solve the puzzle automatically.

## About the 8-Puzzle

The 8-puzzle is a sliding puzzle consisting of a 3×3 grid with 8 numbered tiles and one blank space. The objective is to rearrange the tiles from a given initial configuration to the goal configuration by sliding tiles into the blank space.

**Goal State:**
```
1 2 3
4 5 6
7 8  
```

The 8-puzzle is a classic problem in artificial intelligence used to demonstrate search algorithms. It's a simplified version of the 15-puzzle and has been extensively studied in the field of computer science.

## Features

- Interactive GUI with smooth animations
- Manual puzzle solving by clicking on tiles
- Automatic solving using various search algorithms
- Visual representation of solution paths
- Statistics display for algorithm performance
- Adjustable animation speed
- Random puzzle shuffling

## Search Algorithms

This implementation includes several search algorithms to solve the 8-puzzle automatically:

### 1. Breadth-First Search (BFS)

**Characteristics:**
- Explores all nodes at the present depth before moving to nodes at the next depth level
- Complete: Will find a solution if one exists
- Optimal: Guarantees the shortest path solution (when all step costs are equal)
- Time Complexity: O(b^d) where b is the branching factor and d is the depth of the solution
- Space Complexity: O(b^d) - must store all generated nodes

**Application to 8-Puzzle:**
- Guarantees the shortest solution path
- Can be inefficient for deeper solutions due to exponential memory requirements
- Does not use any heuristic information about the problem

### 2. Depth-First Search (DFS)

**Characteristics:**
- Explores as far as possible along each branch before backtracking
- Not complete for infinite or cyclic state spaces (without cycle detection)
- Not optimal: May find a very long path even when a short one exists
- Time Complexity: O(b^m) where m is the maximum depth of the search tree
- Space Complexity: O(bm) - only needs to store the current path

**Application to 8-Puzzle:**
- May find very inefficient solutions
- Uses less memory than BFS
- Not recommended for 8-puzzle due to potentially long solution paths

### 3. Uniform-Cost Search (UCS)

**Characteristics:**
- Expands nodes in order of their path cost from the start node
- Complete and optimal when step costs are non-negative
- Time and Space Complexity: O(b^(C*/ε)) where C* is the cost of the optimal solution and ε is the minimum action cost

**Application to 8-Puzzle:**
- Since all moves have equal cost in the 8-puzzle, UCS behaves similarly to BFS
- Guarantees the shortest solution path
- Does not use heuristic information

### 4. A* Search (Graph Version)

**Characteristics:**
- Combines UCS with a heuristic function to guide the search
- Uses f(n) = g(n) + h(n) where g(n) is the cost so far and h(n) is the estimated cost to goal
- Complete and optimal when h(n) is admissible (never overestimates) and consistent
- Maintains a closed set to avoid revisiting states (graph search)

**Application to 8-Puzzle:**
- More efficient than uninformed search algorithms
- Uses domain knowledge through heuristics
- The graph version avoids revisiting previously explored states

### 5. A* Search (Tree Version)

**Characteristics:**
- Similar to A* Graph Search but does not maintain a closed set
- May revisit states multiple times (potentially less efficient)
- Sometimes can find solutions faster in certain domains due to less overhead

**Application to 8-Puzzle:**
- Generally less efficient than A* Graph Search for the 8-puzzle
- Included for comparison purposes
- Demonstrates the importance of avoiding repeated states in search

## Heuristic Functions

Two heuristic functions are implemented for the A* algorithms:

### 1. Misplaced Tiles (h1)

- Counts the number of tiles that are not in their goal position
- Admissible: Never overestimates the actual cost
- Simple to compute but not as informative as Manhattan distance

### 2. Manhattan Distance (h2)

- Sums the horizontal and vertical distances of each tile from its goal position
- More informed than the misplaced tiles heuristic
- Still admissible and consistent
- Generally leads to faster solutions with fewer node expansions

## Performance Comparison

When solving the 8-puzzle, the algorithms typically perform as follows (from most to least efficient):

1. A* Graph Search with Manhattan Distance
2. A* Graph Search with Misplaced Tiles
3. A* Tree Search with Manhattan Distance
4. A* Tree Search with Misplaced Tiles
5. BFS/UCS (equivalent for this problem)
6. DFS (often finds longer, suboptimal solutions)

The application displays statistics for each algorithm run:
- Solution Steps: The number of moves in the solution path
- Nodes Expanded: The number of states examined during search
- Max Frontier: The maximum size of the frontier during search

## Implementation Details

The project consists of two main components:

1. **puzzle_solver.py**: Contains the implementation of the search algorithms and heuristic functions
   - `PuzzleNode` class for representing states
   - Heuristic functions (h1, h2)
   - Search algorithm implementations
   - Utility functions for path reconstruction and state validation

2. **puzzle_game.py**: Implements the graphical user interface using Pygame
   - Interactive game board
   - Algorithm selection buttons
   - Solution animation
   - Statistics display

## Usage

To run the application:

```
python puzzle_game.py
```

### Controls:

- Click on a tile adjacent to the blank space to move it
- Click "Shuffle" to randomize the puzzle
- Click "Speed" to cycle through animation speeds
- Click on any algorithm button to solve the puzzle automatically

## Educational Value

This project serves as an educational tool for understanding:

1. Search algorithms in artificial intelligence
2. Heuristic functions and their impact on search efficiency
3. The trade-offs between different search strategies
4. The importance of state representation in problem-solving
5. Basic concepts of algorithm complexity and performance

By comparing the statistics of different algorithms on the same puzzle configuration, users can gain insights into the strengths and weaknesses of each approach.