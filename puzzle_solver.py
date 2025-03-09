import heapq
from collections import deque
from typing import List, Tuple, Callable, Optional

class PuzzleNode:
    """
    A class to represent a state of the n-puzzle game.
    
    Attributes:
        state (List[List[int]]): The puzzle configuration as a 2D grid.
        parent (Optional[PuzzleNode]): The parent node that generated this state.
        g (int): The cost to reach this node from the start node.
        h (int): The heuristic value (estimated cost to goal).
        f (int): The total estimated cost (g + h).
        blank_pos (Tuple[int, int]): The position of the blank space (0).
        n (int): The dimension of the puzzle (3 for 3×3, etc.).
        move (str): The move that led to this state (for path reconstruction).
    """
    def __init__(self, state: List[List[int]], parent=None, g=0, h=0, move=""):
        self.state = [row[:] for row in state]  # Deep copy
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h
        self.n = len(state)
        self.move = move
        
        # Locate the blank (0)
        for i in range(self.n):
            for j in range(self.n):
                if state[i][j] == 0:
                    self.blank_pos = (i, j)
                    break

    def __lt__(self, other):
        """
        Nodes are compared by f = g + h; if ties, compare by h.
        """
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f

    def __eq__(self, other):
        if not isinstance(other, PuzzleNode):
            return False
        return self.state == other.state

    def __hash__(self):
        return hash(str(self.state))

    def __str__(self):
        """
        String representation for printing.
        """
        rows = []
        for row in self.state:
            rows.append(" ".join(str(x) if x != 0 else " " for x in row))
        return "\n".join(rows)

    def get_children(self) -> List['PuzzleNode']:
        """
        Generate children by sliding the blank in up, down, left, or right direction.
        """
        children = []
        moves = [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]
        
        x0, y0 = self.blank_pos
        for dx, dy, move_name in moves:
            nx, ny = x0 + dx, y0 + dy
            if 0 <= nx < self.n and 0 <= ny < self.n:
                new_state = [row[:] for row in self.state]
                # Swap the blank with the adjacent tile
                new_state[x0][y0], new_state[nx][ny] = new_state[nx][ny], new_state[x0][y0]
                child = PuzzleNode(new_state, parent=self, g=self.g + 1, move=move_name)
                children.append(child)
        return children


# -----------------------
# Heuristic Functions
# -----------------------

def h1(state: List[List[int]]) -> int:
    """
    h1: Misplaced tiles heuristic.
    Counts the number of tiles that are not in their goal position.
    """
    n = len(state)
    count = 0
    for i in range(n):
        for j in range(n):
            if state[i][j] == 0:
                continue
            # The tile 'value' should be in position ((value-1)//n, (value-1)%n)
            expected_val = i * n + j + 1
            if expected_val == n * n:
                expected_val = 0  # last tile is blank
            if state[i][j] != expected_val:
                count += 1
    return count

def h2(state: List[List[int]]) -> int:
    """
    h2: Manhattan distance heuristic.
    Sums the Manhattan distances of each tile from its goal position.
    """
    n = len(state)
    total_dist = 0
    for i in range(n):
        for j in range(n):
            value = state[i][j]
            if value == 0:
                continue  # don't count blank
            # For 'value', the goal pos is ((value-1)//n, (value-1)%n)
            goal_x, goal_y = (value - 1) // n, (value - 1) % n
            total_dist += abs(i - goal_x) + abs(j - goal_y)
    return total_dist


# -----------------------
# Utility Functions
# -----------------------

def is_valid_state(state: List[List[int]]) -> bool:
    """
    Checks if 'state' is a valid n×n puzzle, containing 0..n^2-1 exactly once.
    """
    if not isinstance(state, list) or not all(isinstance(row, list) for row in state):
        return False
    n = len(state)
    if any(len(row) != n for row in state):
        return False
    
    flattened = [x for row in state for x in row]
    return sorted(flattened) == list(range(n * n))

def get_goal_state(n: int) -> List[List[int]]:
    """
    Returns the goal state for an n×n puzzle, with 0 in the bottom-right corner.
    """
    goal = []
    for i in range(n):
        row = []
        for j in range(n):
            val = i * n + j + 1
            if val == n * n:
                val = 0
            row.append(val)
        goal.append(row)
    return goal

def reconstruct_path(node: PuzzleNode) -> List[List[List[int]]]:
    """
    Backtracks from the goal node to the start node to retrieve the solution path.
    """
    path = []
    curr = node
    while curr:
        path.append(curr.state)
        curr = curr.parent
    return path[::-1]  # Reverse to get start->goal


# -----------------------
# 1) Breadth-First Search
# -----------------------
def solvePuzzleBFS(
    state: List[List[int]]
) -> Tuple[int, int, int, Optional[List[List[List[int]]]], int]:
    """
    Solve puzzle with BFS. Returns (steps, expanded, max_frontier, path, err).
    """
    if not is_valid_state(state):
        return (0, 0, 0, None, -1)

    n = len(state)
    goal = get_goal_state(n)
    if state == goal:
        return (0, 0, 1, [state], 0)
    
    start_node = PuzzleNode(state)
    queue = deque([start_node])
    visited = {str(start_node.state)}
    
    expanded = 0
    max_frontier = 1

    while queue:
        current = queue.popleft()
        expanded += 1
        
        for child in current.get_children():
            c_str = str(child.state)
            if c_str not in visited:
                visited.add(c_str)
                if child.state == goal:
                    path = reconstruct_path(child)
                    return (len(path) - 1, expanded, max_frontier, path, 0)
                queue.append(child)
        max_frontier = max(max_frontier, len(queue))
    
    return (0, expanded, max_frontier, None, 0)


# -----------------------
# 2) Depth-First Search
# -----------------------
def solvePuzzleDFS(
    state: List[List[int]]
) -> Tuple[int, int, int, Optional[List[List[List[int]]]], int]:
    """
    Solve puzzle with DFS. Returns (steps, expanded, max_frontier, path, err).
    """
    if not is_valid_state(state):
        return (0, 0, 0, None, -1)

    n = len(state)
    goal = get_goal_state(n)
    if state == goal:
        return (0, 0, 1, [state], 0)
    
    start_node = PuzzleNode(state)
    stack = [start_node]
    visited = {str(start_node.state)}
    
    expanded = 0
    max_frontier = 1

    while stack:
        current = stack.pop()
        expanded += 1
        
        for child in reversed(current.get_children()):
            c_str = str(child.state)
            if c_str not in visited:
                visited.add(c_str)
                if child.state == goal:
                    path = reconstruct_path(child)
                    return (len(path) - 1, expanded, max_frontier, path, 0)
                stack.append(child)
        max_frontier = max(max_frontier, len(stack))
    
    return (0, expanded, max_frontier, None, 0)


# -----------------------
# 3) Uniform-Cost Search
# -----------------------
def solvePuzzleUCS(
    state: List[List[int]]
) -> Tuple[int, int, int, Optional[List[List[List[int]]]], int]:
    """
    Solve puzzle with Uniform-Cost Search (Dijkstra). No heuristic.
    Returns (steps, expanded, max_frontier, path, err).
    """
    if not is_valid_state(state):
        return (0, 0, 0, None, -1)

    n = len(state)
    goal = get_goal_state(n)
    if state == goal:
        return (0, 0, 1, [state], 0)
    
    start_node = PuzzleNode(state, g=0, h=0)
    frontier = [(start_node.g, start_node)]
    frontier_set = {str(start_node.state)}
    visited = set()
    
    expanded = 0
    max_frontier = 1

    while frontier:
        _, current = heapq.heappop(frontier)
        frontier_set.remove(str(current.state))
        expanded += 1
        
        if current.state == goal:
            path = reconstruct_path(current)
            return (len(path) - 1, expanded, max_frontier, path, 0)
        
        visited.add(str(current.state))
        
        for child in current.get_children():
            child.h = 0
            child.f = child.g  # same as g for UCS
            c_str = str(child.state)
            if c_str not in visited and c_str not in frontier_set:
                heapq.heappush(frontier, (child.f, child))
                frontier_set.add(c_str)
        max_frontier = max(max_frontier, len(frontier))
    
    return (0, expanded, max_frontier, None, 0)


# -----------------------
# 4) A* Graph Search
# -----------------------
def solvePuzzleAStarGraph(
    state: List[List[int]], 
    heuristic: Callable[[List[List[int]]], int]
) -> Tuple[int, int, int, Optional[List[List[List[int]]]], int]:
    """
    Solve puzzle with A* (graph search), using the provided heuristic.
    We maintain a closed set ('explored') so we do not revisit states we've processed.
    Returns (steps, expanded, max_frontier, path, err).
    """
    if not is_valid_state(state):
        return (0, 0, 0, None, -1)

    n = len(state)
    goal = get_goal_state(n)
    if state == goal:
        return (0, 0, 1, [state], 0)
    
    start_node = PuzzleNode(state, g=0, h=heuristic(state))
    start_node.f = start_node.g + start_node.h

    frontier = [start_node]
    frontier_set = {str(start_node.state)}
    explored = set()
    
    expanded = 0
    max_frontier = 1

    while frontier:
        current = heapq.heappop(frontier)
        frontier_set.remove(str(current.state))
        explored.add(str(current.state))
        expanded += 1
        
        if current.state == goal:
            path = reconstruct_path(current)
            return (len(path) - 1, expanded, max_frontier, path, 0)
        
        for child in current.get_children():
            child.h = heuristic(child.state)
            child.f = child.g + child.h
            c_str = str(child.state)
            if c_str in explored:
                continue
            if c_str not in frontier_set:
                heapq.heappush(frontier, child)
                frontier_set.add(c_str)
            else:
                # Check if new path is better
                for i, node_in_frontier in enumerate(frontier):
                    if str(node_in_frontier.state) == c_str and child.f < node_in_frontier.f:
                        frontier[i] = child
                        heapq.heapify(frontier)
                        break
        max_frontier = max(max_frontier, len(frontier))
    
    return (0, expanded, max_frontier, None, 0)


# -----------------------
# 5) A* Tree Search
# -----------------------
def solvePuzzleAStarTree(
    state: List[List[int]], 
    heuristic: Callable[[List[List[int]]], int]
) -> Tuple[int, int, int, Optional[List[List[List[int]]]], int]:
    """
    Solve puzzle with A* tree search, using the provided heuristic.
    Unlike graph search, tree search does not use a closed set to avoid revisiting states.
    This can lead to higher expansions but can sometimes find solutions faster in certain domains.
    Returns (steps, expanded, max_frontier, path, err).
    """
    if not is_valid_state(state):
        return (0, 0, 0, None, -1)

    n = len(state)
    goal = get_goal_state(n)
    if state == goal:
        return (0, 0, 1, [state], 0)
    
    start_node = PuzzleNode(state, g=0, h=heuristic(state))
    start_node.f = start_node.g + start_node.h

    frontier = []
    heapq.heappush(frontier, start_node)
    frontier_set = {str(start_node.state)}
    
    expanded = 0
    max_frontier = 1

    while frontier:
        current = heapq.heappop(frontier)
        frontier_set.remove(str(current.state))
        expanded += 1
        
        if current.state == goal:
            path = reconstruct_path(current)
            return (len(path) - 1, expanded, max_frontier, path, 0)
        
        for child in current.get_children():
            child.h = heuristic(child.state)
            child.f = child.g + child.h
            c_str = str(child.state)
            
            # No explored set => could revisit states
            if c_str not in frontier_set:
                heapq.heappush(frontier, child)
                frontier_set.add(c_str)
            else:
                # Improve path if found
                for i, node_in_frontier in enumerate(frontier):
                    if str(node_in_frontier.state) == c_str and child.f < node_in_frontier.f:
                        frontier[i] = child
                        heapq.heapify(frontier)
                        break
                        
        max_frontier = max(max_frontier, len(frontier))
    
    return (0, expanded, max_frontier, None, 0)
