import heapq

##############################################################################
# Existing heuristics:
##############################################################################

def h1(state):
    """
    Heuristic 1: Counts the number of tiles not in their correct position, ignoring the blank.
    
    state is a list of lists representing an n x n puzzle.
    """
    n = len(state)
    misplaced = 0
    for r in range(n):
        for c in range(n):
            val = state[r][c]
            if val != 0:  # ignore the blank
                goal_r = val // n
                goal_c = val % n
                if (r != goal_r) or (c != goal_c):
                    misplaced += 1
    return misplaced


def h2(state):
    """
    Heuristic 2: Sums the Manhattan distances of each tile from its correct position,
    ignoring the blank.
    """
    n = len(state)
    total_distance = 0
    
    for r in range(n):
        for c in range(n):
            val = state[r][c]
            if val != 0:
                goal_r = val // n
                goal_c = val % n
                total_distance += abs(r - goal_r) + abs(c - goal_c)
    
    return total_distance


def h3(state):
    """
    Heuristic 3: Linear Conflict + Manhattan Distance
    
    Returns a value >= the standard Manhattan distance, adding +2 for each linear conflict.
    """
    n = len(state)
    manhattan_sum = 0
    row_conflicts = 0
    col_conflicts = 0

    # Prepare structures to track tiles in their correct row/column
    row_positions = [[] for _ in range(n)]  # row_positions[r] = list of (tile_value, current_col)
    col_positions = [[] for _ in range(n)]  # col_positions[c] = list of (tile_value, current_row)

    # 1) Sum of Manhattan distances, plus gather row/column info
    for r in range(n):
        for c in range(n):
            val = state[r][c]
            if val != 0:
                goal_r = val // n
                goal_c = val % n
                manhattan_sum += abs(r - goal_r) + abs(c - goal_c)
                
                # If tile belongs in row goal_r and it's currently at row r
                if goal_r == r:
                    row_positions[r].append((val, c))
                # If tile belongs in column goal_c and it's currently at column c
                if goal_c == c:
                    col_positions[c].append((val, r))

    # 2) Detect row conflicts
    for r in range(n):
        row_data = row_positions[r]
        # Compare all pairs in row_data
        for i in range(len(row_data)):
            for j in range(i + 1, len(row_data)):
                val_i, col_i = row_data[i]
                val_j, col_j = row_data[j]
                # Conflict if tile i has a larger value than tile j but is to the left
                if val_i > val_j and col_i < col_j:
                    row_conflicts += 1

    # 3) Detect column conflicts
    for c in range(n):
        col_data = col_positions[c]
        # Compare all pairs in col_data
        for i in range(len(col_data)):
            for j in range(i + 1, len(col_data)):
                val_i, row_i = col_data[i]
                val_j, row_j = col_data[j]
                # Conflict if tile i has a larger value than tile j but is above
                if val_i > val_j and row_i < row_j:
                    col_conflicts += 1

    # 4) Each conflict adds +2 to the total
    return manhattan_sum + 2 * (row_conflicts + col_conflicts)

# Package all heuristics into a list so you can select them by index
heuristics = [h1, h2, h3]

##############################################################################
# PuzzleNode class:
##############################################################################

class PuzzleNode:
    """
    Represents a node in the search tree for an n x n sliding puzzle.
    """
    def __init__(self, state, n, parent=None, g_cost=0, h_cost=0):
        self.state = state
        self.n = n
        self.parent = parent
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost

    def __str__(self):
        return "\n".join(" ".join(str(x) if x != 0 else "_" for x in row)
                         for row in self.state)

    def generate_children(self):
        children = []
        blank_r, blank_c = None, None
        
        # Find blank tile
        for i in range(self.n):
            for j in range(self.n):
                if self.state[i][j] == 0:
                    blank_r, blank_c = i, j
                    break
            if blank_r is not None:
                break
        
        # Possible moves: up, down, left, right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in moves:
            new_r = blank_r + dr
            new_c = blank_c + dc
            # Check boundaries
            if 0 <= new_r < self.n and 0 <= new_c < self.n:
                new_state = [row[:] for row in self.state]
                new_state[blank_r][blank_c], new_state[new_r][new_c] = (
                    new_state[new_r][new_c],
                    new_state[blank_r][blank_c]
                )
                child = PuzzleNode(new_state, self.n, parent=self,
                                   g_cost=self.g_cost + 1)
                children.append(child)
        
        return children

##############################################################################
# A* puzzle solver that can use any of the three heuristics by index
##############################################################################

def solvePuzzle(state, heuristic_index=0):
    """
    Perform A* search on an n x n puzzle using one of the three heuristics (h1, h2, h3).
    
    Parameters
    ----------
    state          : list of lists
                    The initial scrambled puzzle state, with 0 as the blank.
    heuristic_index: int
                    0 -> h1 (misplaced tiles)
                    1 -> h2 (Manhattan distance)
                    2 -> h3 (Linear conflict + Manhattan)
    
    Returns
    -------
    steps        : int
                   Number of moves in the optimal path (excluding the initial state)
    expansions   : int
                   Number of nodes expanded
    max_frontier : int
                   The maximum size the frontier reached during the search
    solution_path: list of list of lists
                   The optimal path from start state to goal state
    err          : int
                   Error code: 0 = no error, -1 = invalid puzzle, -2 = puzzle unsolvable
    """
    
    # Validate dimension
    if not state or any(len(row) != len(state) for row in state):
        return (0, 0, 0, None, -1)

    n = len(state)
    
    # Validate content
    required = set(range(n*n))
    found = set()
    for row in state:
        found.update(row)
    if found != required:
        return (0, 0, 0, None, -1)

    # Check solvability (if you have an is_solvable function)
    if not is_solvable(state):
        return (0, 0, 0, None, -2)

    # Select heuristic by index
    if heuristic_index < 0 or heuristic_index >= len(heuristics):
        heuristic_index = 0  # default to h1 if out of range
    heuristic_func = heuristics[heuristic_index]

    # Prepare the start node
    start_node = PuzzleNode(state, n)
    start_node.h_cost = heuristic_func(state)
    start_node.f_cost = start_node.g_cost + start_node.h_cost

    # Priority queue for A* (f_cost, tie_breaker, node)
    frontier = []
    tie_breaker = 0
    heapq.heappush(frontier, (start_node.f_cost, tie_breaker, start_node))

    visited = set()
    visited.add(state_to_tuple(state))

    expansions = 0
    max_frontier = 1

    while frontier:
        f_cost, _, current = heapq.heappop(frontier)

        # Check for goal
        if is_goal(current.state):
            path = reconstruct_path(current)
            steps = len(path) - 1
            return (steps, expansions, max_frontier, path, 0)

        expansions += 1

        # Expand children
        children = current.generate_children()
        for child in children:
            child_state_tuple = state_to_tuple(child.state)
            if child_state_tuple not in visited:
                # Evaluate heuristic
                child.h_cost = heuristic_func(child.state)
                child.f_cost = child.g_cost + child.h_cost

                tie_breaker += 1
                heapq.heappush(frontier, (child.f_cost, tie_breaker, child))
                visited.add(child_state_tuple)

        max_frontier = max(max_frontier, len(frontier))

    # If we exhaust the frontier, puzzle is unsolvable (theoretically shouldn't happen
    # if is_solvable returned True)
    return (0, expansions, max_frontier, None, -2)

##############################################################################
# Helper functions
##############################################################################

def is_goal(state):
    """
    Checks if state is in the goal configuration:
    0 in top-left, then 1..n^2-1 in row-major order.
    """
    n = len(state)
    flat = []
    for row in state:
        flat.extend(row)
    return flat == list(range(n*n))

def state_to_tuple(state):
    """
    Convert 2D list state to a tuple of tuples for hashing in visited sets.
    """
    return tuple(tuple(row) for row in state)

def reconstruct_path(node):
    """
    Reconstruct the path from the start to the current node by following 'parent' pointers.
    """
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return list(reversed(path))

def is_solvable(state):
    """
    Determines if the puzzle is solvable by counting inversions and considering blank row.
    
    For n odd:
        puzzle is solvable if #inversions is even.
    For n even:
        puzzle is solvable if:
          - blank is on an even row counting from bottom & #inversions is odd, OR
          - blank is on an odd row counting from bottom & #inversions is even.
    """
    n = len(state)
    flat = []
    blank_row = 0  # track the row of blank (0)

    # Flatten
    for i in range(n):
        for j in range(n):
            val = state[i][j]
            if val == 0:
                blank_row = i
            flat.append(val)

    # Count inversions (ignoring 0)
    inversions = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] != 0 and flat[j] != 0 and flat[i] > flat[j]:
                inversions += 1

    if n % 2 == 1:
        return (inversions % 2 == 0)
    else:
        blank_row_from_bottom = (n - 1) - blank_row
        return ((blank_row_from_bottom % 2 == 0) ^ (inversions % 2 == 0))
