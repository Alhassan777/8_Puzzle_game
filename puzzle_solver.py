import heapq
from collections import deque

##############################################################################
# 1) Heuristics: h1, h2, h3
##############################################################################

def h1(state):
    """
    Heuristic 1: Counts the number of tiles not in their correct position, ignoring the blank.
    """
    n = len(state)
    misplaced = 0
    for r in range(n):
        for c in range(n):
            val = state[r][c]
            if val != 0:  # ignore blank
                goal_r = val // n
                goal_c = val % n
                if r != goal_r or c != goal_c:
                    misplaced += 1
    return misplaced


def h2(state):
    """
    Heuristic 2: Sum of Manhattan distances of each tile from its correct position, ignoring the blank.
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

    Adds +2 for every "linear conflict" pair of tiles that are in their correct row (or column)
    but reversed relative to their goal order. This yields a heuristic >= plain Manhattan distance.
    """
    n = len(state)
    manhattan_sum = 0
    row_conflicts = 0
    col_conflicts = 0

    row_positions = [[] for _ in range(n)]  # row_positions[r] = [(tile_val, current_col), ...]
    col_positions = [[] for _ in range(n)]  # col_positions[c] = [(tile_val, current_row), ...]

    # Compute base Manhattan + collect row/col data
    for r in range(n):
        for c in range(n):
            val = state[r][c]
            if val != 0:
                goal_r = val // n
                goal_c = val % n
                manhattan_sum += abs(r - goal_r) + abs(c - goal_c)
                
                # If tile is in its goal row, track for row conflict
                if goal_r == r:
                    row_positions[r].append((val, c))
                # If tile is in its goal column, track for col conflict
                if goal_c == c:
                    col_positions[c].append((val, r))

    # Check row conflicts
    for r in range(n):
        row_data = row_positions[r]
        for i in range(len(row_data)):
            for j in range(i+1, len(row_data)):
                val_i, col_i = row_data[i]
                val_j, col_j = row_data[j]
                if val_i > val_j and col_i < col_j:
                    row_conflicts += 1

    # Check column conflicts
    for c in range(n):
        col_data = col_positions[c]
        for i in range(len(col_data)):
            for j in range(i+1, len(col_data)):
                val_i, row_i = col_data[i]
                val_j, row_j = col_data[j]
                if val_i > val_j and row_i < row_j:
                    col_conflicts += 1

    return manhattan_sum + 2 * (row_conflicts + col_conflicts)

heuristics = [h1, h2, h3]

##############################################################################
# 2) PuzzleNode class + small utility
##############################################################################

class PuzzleNode:
    """
    Node class for n x n sliding puzzle solvers (BFS, DFS, UCS, A*, etc.).
    """
    def __init__(self, state, n, parent=None, cost_so_far=0):
        self.state = state
        self.n = n
        self.parent = parent
        self.cost_so_far = cost_so_far  # g-cost for UCS / A*

        # For A*, we'll store h_cost and f_cost
        self.h_cost = 0
        self.f_cost = 0
    
    def generate_children(self):
        children = []
        blank_r, blank_c = None, None
        for i in range(self.n):
            for j in range(self.n):
                if self.state[i][j] == 0:
                    blank_r, blank_c = i, j
                    break
            if blank_r is not None:
                break
        
        moves = [(-1,0), (1,0), (0,-1), (0,1)]
        for dr, dc in moves:
            nr = blank_r + dr
            nc = blank_c + dc
            if 0 <= nr < self.n and 0 <= nc < self.n:
                # Copy current state
                new_state = [row[:] for row in self.state]
                # Swap
                new_state[blank_r][blank_c], new_state[nr][nc] = new_state[nr][nc], new_state[blank_r][blank_c]
                
                child = PuzzleNode(new_state, self.n, parent=self, cost_so_far=self.cost_so_far + 1)
                children.append(child)
        return children

def is_goal(state):
    """
    Checks if 'state' is the goal configuration: 0 in top-left, then 1..n^2-1 in row-major order.
    """
    # Flatten
    n = len(state)
    flat = [tile for row in state for tile in row]
    # Expect 1..n^2-1, then 0
    return flat == list(range(1, n*n)) + [0]

def state_to_tuple(state):
    """Convert a 2D list 'state' into a tuple of tuples for hashing."""
    return tuple(tuple(r) for r in state)

def reconstruct_path(node):
    """
    Reconstruct path from a goal node back to the root by following parent pointers.
    """
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return list(reversed(path))

##############################################################################
# 3) Solvability Checker
##############################################################################

def is_solvable(state):
    """
    For n odd:
       puzzle is solvable if #inversions is even.
    For n even:
       puzzle is solvable if:
         - blank is on an even row counting from bottom & #inversions is odd, OR
         - blank is on an odd row  counting from bottom & #inversions is even.
    """
    n = len(state)
    flat = []
    blank_row = 0
    for i in range(n):
        for j in range(n):
            val = state[i][j]
            if val == 0:
                blank_row = i
            flat.append(val)
    
    # Count inversions
    inversions = 0
    for i in range(len(flat)):
        for j in range(i+1, len(flat)):
            if flat[i] != 0 and flat[j] != 0 and flat[i] > flat[j]:
                inversions += 1
    
    if n % 2 == 1:
        return (inversions % 2 == 0)
    else:
        blank_row_from_bottom = (n - 1) - blank_row
        return ((blank_row_from_bottom % 2 == 0) ^ (inversions % 2 == 0))

##############################################################################
# 4) BFS, DFS, UCS
##############################################################################

def solvePuzzleBFS(state):
    """
    Standard Breadth-First Search on the puzzle, ignoring costs/heuristics.
    Returns (steps, expanded, max_frontier, path, err).
    """
    # Validate
    if not state or any(len(row) != len(state) for row in state):
        return (0, 0, 0, None, -1)
    n = len(state)
    required = set(range(n*n))
    found = {x for row in state for x in row}
    if found != required:
        return (0, 0, 0, None, -1)
    if not is_solvable(state):
        return (0, 0, 0, None, -2)
    
    start = PuzzleNode(state, n)
    queue = deque([start])
    visited = set([state_to_tuple(state)])
    expanded = 0
    max_frontier = 1
    
    while queue:
        current = queue.popleft()
        if is_goal(current.state):
            path = reconstruct_path(current)
            steps = len(path) - 1
            return (steps, expanded, max_frontier, path, 0)
        
        expanded += 1
        children = current.generate_children()
        for child in children:
            st = state_to_tuple(child.state)
            if st not in visited:
                visited.add(st)
                queue.append(child)
        max_frontier = max(max_frontier, len(queue))
    
    return (0, expanded, max_frontier, None, -2)

def solvePuzzleDFS(state):
    """
    Depth-First Search on the puzzle. 
    Potentially huge search trees, but included for completeness.
    """
    if not state or any(len(row) != len(state) for row in state):
        return (0, 0, 0, None, -1)
    n = len(state)
    required = set(range(n*n))
    found = {x for row in state for x in row}
    if found != required:
        return (0, 0, 0, None, -1)
    if not is_solvable(state):
        return (0, 0, 0, None, -2)
    
    start = PuzzleNode(state, n)
    stack = [start]
    visited = set([state_to_tuple(state)])
    expanded = 0
    max_frontier = 1
    
    while stack:
        current = stack.pop()
        if is_goal(current.state):
            path = reconstruct_path(current)
            steps = len(path) - 1
            return (steps, expanded, max_frontier, path, 0)
        
        expanded += 1
        children = current.generate_children()
        for child in children:
            st = state_to_tuple(child.state)
            if st not in visited:
                visited.add(st)
                stack.append(child)
        max_frontier = max(max_frontier, len(stack))
    
    return (0, expanded, max_frontier, None, -2)

def solvePuzzleUCS(state):
    """
    Uniform-Cost Search, ignoring any heuristic. Expands nodes in order of cost_so_far.
    """
    if not state or any(len(row) != len(state) for row in state):
        return (0, 0, 0, None, -1)
    n = len(state)
    required = set(range(n*n))
    found = {x for row in state for x in row}
    if found != required:
        return (0, 0, 0, None, -1)
    if not is_solvable(state):
        return (0, 0, 0, None, -2)
    
    start = PuzzleNode(state, n)
    frontier = []
    tie_break = 0
    # Priority queue keyed by cost_so_far
    heapq.heappush(frontier, (start.cost_so_far, tie_break, start))
    visited = set([state_to_tuple(state)])
    
    expanded = 0
    max_frontier = 1
    
    while frontier:
        cost, _, current = heapq.heappop(frontier)
        if is_goal(current.state):
            path = reconstruct_path(current)
            steps = len(path) - 1
            return (steps, expanded, max_frontier, path, 0)
        
        expanded += 1
        for child in current.generate_children():
            st = state_to_tuple(child.state)
            if st not in visited:
                visited.add(st)
                tie_break += 1
                heapq.heappush(frontier, (child.cost_so_far, tie_break, child))
        
        max_frontier = max(max_frontier, len(frontier))
    
    return (0, expanded, max_frontier, None, -2)

##############################################################################
# 5) A* (Graph Search) and A* (Tree Search) with selectable heuristic
##############################################################################

def solvePuzzleAStarGraph(state, heuristic_index=1):
    """
    A* Graph Search. 
    We store visited states so we do not revisit them repeatedly. 
    heuristic_index picks which heuristic to use:
       0 -> h1, 1 -> h2, 2 -> h3
    """
    if not state or any(len(row) != len(state) for row in state):
        return (0, 0, 0, None, -1)
    n = len(state)
    required = set(range(n*n))
    found = {x for row in state for x in row}
    if found != required:
        return (0, 0, 0, None, -1)
    if not is_solvable(state):
        return (0, 0, 0, None, -2)
    
    # Pick the heuristic function
    if heuristic_index < 0 or heuristic_index >= len(heuristics):
        heuristic_index = 1  # default to h2 (Manhattan) if out of range
    heuristic_func = heuristics[heuristic_index]
    
    start = PuzzleNode(state, n)
    start.h_cost = heuristic_func(state)
    start.f_cost = start.cost_so_far + start.h_cost

    frontier = []
    tie_break = 0
    heapq.heappush(frontier, (start.f_cost, tie_break, start))
    visited = set([state_to_tuple(state)])
    
    expanded = 0
    max_frontier = 1
    
    while frontier:
        f_val, _, current = heapq.heappop(frontier)
        
        if is_goal(current.state):
            path = reconstruct_path(current)
            steps = len(path) - 1
            return (steps, expanded, max_frontier, path, 0)
        
        expanded += 1
        
        # Expand children
        for child in current.generate_children():
            st = state_to_tuple(child.state)
            if st not in visited:
                visited.add(st)
                child.h_cost = heuristic_func(child.state)
                child.f_cost = child.cost_so_far + child.h_cost
                tie_break += 1
                heapq.heappush(frontier, (child.f_cost, tie_break, child))
        
        max_frontier = max(max_frontier, len(frontier))
    
    return (0, expanded, max_frontier, None, -2)


def solvePuzzleAStarTree(state, heuristic_index=1):
    """
    A* Tree Search. 
    Does not track visited states, so we might re-expand them if encountered on a different path.
    heuristic_index picks which heuristic to use:
       0 -> h1, 1 -> h2, 2 -> h3
    """
    if not state or any(len(row) != len(state) for row in state):
        return (0, 0, 0, None, -1)
    n = len(state)
    required = set(range(n*n))
    found = {x for row in state for x in row}
    if found != required:
        return (0, 0, 0, None, -1)
    if not is_solvable(state):
        return (0, 0, 0, None, -2)
    
    # Pick the heuristic function
    if heuristic_index < 0 or heuristic_index >= len(heuristics):
        heuristic_index = 1  # default to h2 (Manhattan)
    heuristic_func = heuristics[heuristic_index]
    
    start = PuzzleNode(state, n)
    start.h_cost = heuristic_func(state)
    start.f_cost = start.cost_so_far + start.h_cost
    
    frontier = []
    tie_break = 0
    heapq.heappush(frontier, (start.f_cost, tie_break, start))
    
    expanded = 0
    max_frontier = 1

    while frontier:
        f_val, _, current = heapq.heappop(frontier)
        
        if is_goal(current.state):
            path = reconstruct_path(current)
            steps = len(path) - 1
            return (steps, expanded, max_frontier, path, 0)
        
        expanded += 1
        
        # Generate children
        for child in current.generate_children():
            child.h_cost = heuristic_func(child.state)
            child.f_cost = child.cost_so_far + child.h_cost
            tie_break += 1
            heapq.heappush(frontier, (child.f_cost, tie_break, child))
        
        max_frontier = max(max_frontier, len(frontier))
    
    return (0, expanded, max_frontier, None, -2)
