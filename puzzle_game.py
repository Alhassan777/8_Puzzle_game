import pygame
import sys
import random
import time
from puzzle_solver import (
    PuzzleNode, 
    solvePuzzleBFS,
    solvePuzzleDFS,
    solvePuzzleUCS,
    solvePuzzleAStarGraph,
    solvePuzzleAStarTree, 
    h1, h2, h3  # If you want direct access to heuristics
)


# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 750  # Extra height for buttons and info
TILE_SIZE = 100
GRID_SIZE = 3  # 3×3 grid for the 8-puzzle
GRID_WIDTH = GRID_SIZE * TILE_SIZE
GRID_HEIGHT = GRID_SIZE * TILE_SIZE
GRID_OFFSET_X = (WIDTH - GRID_WIDTH) // 2
GRID_OFFSET_Y = 50
BUTTON_HEIGHT = 40
BUTTON_MARGIN = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_BLUE = (0, 0, 139)
BG_COLOR = (240, 248, 255)       # Light blue background
TILE_COLOR = (100, 149, 237)     # Cornflower blue for tiles
TILE_HIGHLIGHT = (144, 238, 144) # Light green for correct tiles

# Button colors - distinct colors for different button types
BUTTON_COLOR = (176, 196, 222)   # Light steel blue for buttons (default)
BUTTON_HOVER = (135, 206, 250)   # Light sky blue for button hover (default)

# New distinct button colors
ALGORITHM_BTN_COLOR = (75, 0, 130)      # Indigo for algorithm buttons
ALGORITHM_BTN_HOVER = (138, 43, 226)    # BlueViolet for algorithm hover
SHUFFLE_BTN_COLOR = (0, 128, 0)         # Green for shuffle button
SHUFFLE_BTN_HOVER = (50, 205, 50)       # LimeGreen for shuffle hover
SPEED_BTN_COLOR = (220, 20, 60)         # Crimson for speed button
SPEED_BTN_HOVER = (255, 69, 0)          # OrangeRed for speed hover

# Tile colors - distinct color for each number
TILE_COLORS = {
    1: (46, 204, 113),    # Emerald Green
    2: (52, 152, 219),     # Blue
    3: (230, 126, 34),     # Orange
    4: (52, 73, 94),       # Dark Blue
    5: (142, 68, 173),     # Purple
    6: (41, 128, 185),     # Light Blue
    7: (192, 57, 43),      # Red
    8: (39, 174, 96)       # Green
}

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('8-Puzzle Game')

# Fonts
font = pygame.font.SysFont('Arial', 36)
small_font = pygame.font.SysFont('Arial', 20)
info_font = pygame.font.SysFont('Arial', 18)

class PuzzleGame:
    def __init__(self):
        # Default puzzle state (goal)
        self.state = [[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 0]]
        self.goal_state = [[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 0]]
        self.blank_pos = (2, 2)
        self.moves = 0
        
        # For holding a solution path
        self.solution_path = None
        self.solution_index = 0
        self.solving = False
        self.solution_stats = None
        
        # Animation speed (milliseconds)
        self.animation_speed = 500
        self.last_move_time = 0
        
        # Highlight the last moved tile
        self.last_moved_tile = None
        
        # Shuffle on startup
        self.shuffle(30)
    
    def shuffle(self, moves=30):
        """Shuffle the puzzle by making random valid moves."""
        self.state = [[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 0]]
        self.blank_pos = (2, 2)
        self.moves = 0
        self.solution_path = None
        self.solution_index = 0
        self.solving = False
        self.solution_stats = None
        
        for _ in range(moves):
            possible_moves = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx = self.blank_pos[0] + dx
                ny = self.blank_pos[1] + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    possible_moves.append((nx, ny))
            if possible_moves:
                chosen = random.choice(possible_moves)
                self.move_tile(chosen[0], chosen[1], increment_move=False)
    
    def move_tile(self, row, col, increment_move=True):
        """Slide a tile if adjacent to the blank space."""
        if self.is_adjacent(row, col, self.blank_pos[0], self.blank_pos[1]):
            self.last_moved_tile = self.state[row][col]
            self.state[self.blank_pos[0]][self.blank_pos[1]] = self.state[row][col]
            self.state[row][col] = 0
            self.blank_pos = (row, col)
            if increment_move:
                self.moves += 1
            return True
        return False
    
    def is_adjacent(self, r1, c1, r2, c2):
        return (abs(r1 - r2) == 1 and c1 == c2) or \
               (abs(c1 - c2) == 1 and r1 == r2)
    
    def is_solved(self):
        return self.state == self.goal_state
    
    def solve(self, algorithm: str):
        """
        Solve the puzzle with one of the following:
          - BFS
          - DFS
          - UCS
          - A* Graph (Manhattan)
          - A* Tree (Manhattan)
        """
        # Reset old solution
        self.solution_path = None
        self.solution_index = 0
        self.solving = False
        self.solution_stats = None
        
        if algorithm == "BFS":
            steps, expanded, max_frontier, path, err = solvePuzzleBFS(self.state)
            self._save_solution(steps, expanded, max_frontier, path, err, "BFS", "N/A")
        
        elif algorithm == "DFS":
            steps, expanded, max_frontier, path, err = solvePuzzleDFS(self.state)
            self._save_solution(steps, expanded, max_frontier, path, err, "DFS", "N/A")
        
        elif algorithm == "UCS":
            steps, expanded, max_frontier, path, err = solvePuzzleUCS(self.state)
            self._save_solution(steps, expanded, max_frontier, path, err, "UCS", "N/A")
        
        elif algorithm == "A* Graph":
            # Always use Manhattan distance for A*
            steps, expanded, max_frontier, path, err = solvePuzzleAStarGraph(self.state, h2)
            self._save_solution(steps, expanded, max_frontier, path, err, 
                                "A* Graph", "Manhattan Distance")
        
        elif algorithm == "A* Tree":
            # Always use Manhattan distance for A*
            steps, expanded, max_frontier, path, err = solvePuzzleAStarTree(self.state, h2)
            self._save_solution(steps, expanded, max_frontier, path, err, 
                                "A* Tree", "Manhattan Distance")
    
    def _save_solution(self, steps, expanded, max_frontier, path, err, algo_name, heuristic_name):
        """Stores the solution details (if any) for display/animation."""
        if err == 0 and path is not None:
            self.solution_path = path
            self.solution_index = 0
            self.solving = True
            self.solution_stats = {
                'steps': steps,
                'expanded': expanded,
                'max_frontier': max_frontier,
                'algorithm': algo_name,
                'heuristic': heuristic_name
            }
        else:
            self.solution_path = None
            self.solution_index = 0
            self.solving = False
            self.solution_stats = None

    def step_solution(self):
        """Advance one step in the solution path if solving."""
        if self.solving and self.solution_path and self.solution_index < len(self.solution_path) - 1:
            now = pygame.time.get_ticks()
            if now - self.last_move_time > self.animation_speed:
                self.solution_index += 1
                self.state = [row[:] for row in self.solution_path[self.solution_index]]
                
                # Update blank position
                for i in range(GRID_SIZE):
                    for j in range(GRID_SIZE):
                        if self.state[i][j] == 0:
                            self.blank_pos = (i, j)
                            break
                self.last_move_time = now
                if self.solution_index == len(self.solution_path) - 1:
                    self.solving = False

def draw_tile(surface, value, row, col, highlight=False, game=None):
    x = GRID_OFFSET_X + col * TILE_SIZE
    y = GRID_OFFSET_Y + row * TILE_SIZE
    
    # If blank
    if value == 0:
        pygame.draw.rect(surface, WHITE, (x, y, TILE_SIZE, TILE_SIZE))
        return
    
    # Choose tile color
    if highlight:
        base_color = TILE_HIGHLIGHT
    elif game and value == game.last_moved_tile:
        base_color = (255, 165, 0)  # Orange for last moved tile
    else:
        base_color = TILE_COLOR
    
    # Gradient fill
    for i in range(TILE_SIZE):
        grad_factor = 0.7 + (i / TILE_SIZE) * 0.3
        grad_color = tuple(min(255, int(c * grad_factor)) for c in base_color)
        pygame.draw.line(surface, grad_color, (x, y + i), (x + TILE_SIZE - 1, y + i))
    
    # Tile border and shadows
    pygame.draw.rect(surface, BLACK, (x, y, TILE_SIZE, TILE_SIZE), 2)
    pygame.draw.line(surface, (50, 50, 50), 
                     (x, y + TILE_SIZE - 1), 
                     (x + TILE_SIZE - 1, y + TILE_SIZE - 1), 2)
    pygame.draw.line(surface, (50, 50, 50), 
                     (x + TILE_SIZE - 1, y), 
                     (x + TILE_SIZE - 1, y + TILE_SIZE - 1), 2)

    # Tile label with shadow
    shadow_offset = 1
    shadow_text = font.render(str(value), True, (50, 50, 50))
    text = font.render(str(value), True, WHITE)
    text_rect = text.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
    shadow_rect = shadow_text.get_rect(
        center=(text_rect.centerx + shadow_offset, text_rect.centery + shadow_offset)
    )
    
    surface.blit(shadow_text, shadow_rect)
    surface.blit(text, text_rect)

def draw_button(surface, text, x, y, w, h, color, hover_color, action=None):
    mouse_pos = pygame.mouse.get_pos()
    clicked = pygame.mouse.get_pressed()[0]
    
    button_rect = pygame.Rect(x, y, w, h)
    hovering = button_rect.collidepoint(mouse_pos)
    
    # Gradient background
    if hovering:
        for i in range(h):
            grad_factor = 0.8 + (i / h) * 0.2
            grad_color = tuple(min(255, int(c * grad_factor)) for c in hover_color)
            pygame.draw.line(surface, grad_color, (x, y + i), (x + w - 1, y + i))
        if clicked and action:
            action()
    else:
        for i in range(h):
            grad_factor = 0.8 + (i / h) * 0.2
            grad_color = tuple(min(255, int(c * grad_factor)) for c in color)
            pygame.draw.line(surface, grad_color, (x, y + i), (x + w - 1, y + i))
    
    # 2D border
    pygame.draw.rect(surface, BLACK, button_rect, 2)
    if not hovering:
        pygame.draw.line(surface, (50, 50, 50), (x, y + h - 1), (x + w - 1, y + h - 1), 2)
        pygame.draw.line(surface, (50, 50, 50), (x + w - 1, y), (x + w - 1, y + h - 1), 2)
    
    # Button text + shadow
    shadow_offset = 1
    shadow_text = small_font.render(text, True, (50, 50, 50))
    txt_surf = small_font.render(text, True, BLACK)
    txt_rect = txt_surf.get_rect(center=(x + w // 2, y + h // 2))
    shadow_rect = shadow_text.get_rect(center=(
        txt_rect.centerx + shadow_offset, 
        txt_rect.centery + shadow_offset
    ))
    
    surface.blit(shadow_text, shadow_rect)
    surface.blit(txt_surf, txt_rect)

def draw_info(surface, game):
    # Title with drop shadow
    shadow_offset = 2
    title_shadow = font.render("8-Puzzle Game", True, (100, 100, 100))
    title = font.render("8-Puzzle Game", True, DARK_BLUE)
    title_rect = title.get_rect(center=(WIDTH // 2, 25))
    shadow_rect = title_shadow.get_rect(center=(WIDTH // 2 + shadow_offset, 25 + shadow_offset))
    surface.blit(title_shadow, shadow_rect)
    surface.blit(title, title_rect)
    
    # Info panel background
    info_bg = pygame.Rect(10, HEIGHT - 90, WIDTH - 20, 80)
    pygame.draw.rect(surface, (230, 240, 255), info_bg, border_radius=5)
    pygame.draw.rect(surface, (180, 200, 220), info_bg, 2, border_radius=5)
    
    # Moves count
    moves_text = info_font.render(f"Moves: {game.moves}", True, DARK_BLUE)
    surface.blit(moves_text, (20, HEIGHT - 80))
    
    # If a solution is found, show algorithm stats
    if game.solution_stats:
        stats_y = HEIGHT - 60
        alg_text = info_font.render(
            f"Algorithm: {game.solution_stats['algorithm']} "
            f"(Heuristic: {game.solution_stats['heuristic']})", 
            True, DARK_BLUE
        )
        surface.blit(alg_text, (20, stats_y))
        stats_y += 20
        
        stats = [
            f"Solution Steps: {game.solution_stats['steps']}",
            f"Nodes Expanded: {game.solution_stats['expanded']}",
            f"Max Frontier: {game.solution_stats['max_frontier']}"
        ]
        for line in stats:
            line_text = info_font.render(line, True, DARK_BLUE)
            surface.blit(line_text, (20, stats_y))
            stats_y += 20

def main():
    game = PuzzleGame()
    clock = pygame.time.Clock()
    running = True
    
    # Only five algorithms
    algorithms = [
        "BFS", 
        "DFS", 
        "UCS", 
        "A* Graph", 
        "A* Tree"
    ]
    
    # Two columns of buttons, plus top row for Shuffle/Speed
    button_width = (WIDTH - 3 * BUTTON_MARGIN) // 2
    button_x1 = BUTTON_MARGIN
    button_x2 = 2 * BUTTON_MARGIN + button_width
    
    buttons_start_y = GRID_OFFSET_Y + GRID_HEIGHT + BUTTON_MARGIN
    row_step = BUTTON_HEIGHT + BUTTON_MARGIN
    
    while running:
        screen.fill(BG_COLOR)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not game.solving:
                # User clicked on the puzzle area?
                mx, my = pygame.mouse.get_pos()
                if (GRID_OFFSET_X <= mx < GRID_OFFSET_X + GRID_WIDTH) and \
                   (GRID_OFFSET_Y <= my < GRID_OFFSET_Y + GRID_HEIGHT):
                    col = (mx - GRID_OFFSET_X) // TILE_SIZE
                    row = (my - GRID_OFFSET_Y) // TILE_SIZE
                    game.move_tile(row, col, increment_move=True)
        
        # Puzzle area background
        grid_bg = pygame.Rect(
            GRID_OFFSET_X - 10, GRID_OFFSET_Y - 10, 
            GRID_WIDTH + 20, GRID_HEIGHT + 20
        )
        pygame.draw.rect(screen, (220, 230, 240), grid_bg, border_radius=10)
        pygame.draw.rect(screen, (180, 200, 220), grid_bg, 2, border_radius=10)
        
        # Draw puzzle
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                val = game.state[r][c]
                correct = (val != 0 and val == game.goal_state[r][c])
                draw_tile(screen, val, r, c, highlight=correct, game=game)
        
        # Row 0: Shuffle + Speed
        draw_button(
            screen, "Shuffle", 
            button_x1, buttons_start_y, 
            button_width, BUTTON_HEIGHT, 
            BUTTON_COLOR, BUTTON_HOVER,
            action=lambda: game.shuffle(30)
        )
        
        speed_label = ("Speed: Fast" if game.animation_speed < 300 else
                       "Speed: Medium" if game.animation_speed < 700 else
                       "Speed: Slow")
        draw_button(
            screen, speed_label, 
            button_x2, buttons_start_y, 
            button_width, BUTTON_HEIGHT, 
            BUTTON_COLOR, BUTTON_HOVER,
            action=lambda: setattr(
                game, 'animation_speed', 
                (game.animation_speed + 300) % 900 + 100
            )
        )
        
        # Remaining rows: BFS, DFS, UCS, A* Graph, A* Tree
        for i, alg in enumerate(algorithms):
            row_index = i // 2 + 1  # offset by 1 for top row
            col_index = i % 2
            bx = button_x1 if col_index == 0 else button_x2
            by = buttons_start_y + row_index * row_step
            
            draw_button(
                screen, alg,
                bx, by,
                button_width, BUTTON_HEIGHT,
                BUTTON_COLOR, BUTTON_HOVER,
                action=lambda a=alg: game.solve(a)
            )
        
        # Info panel
        draw_info(screen, game)
        
        # Step the solution if we are currently solving
        if game.solving:
            game.step_solution()
        
        # If solved
        if game.is_solved() and not game.solving:
            solved_bg = pygame.Rect(WIDTH // 4, HEIGHT - 145, WIDTH // 2, 40)
            pygame.draw.rect(screen, (220, 255, 220), solved_bg, border_radius=10)
            pygame.draw.rect(screen, (100, 200, 100), solved_bg, 2, border_radius=10)
            
            solved_text = font.render("Puzzle Solved!", True, (0, 100, 0))
            text_rect = solved_text.get_rect(center=(WIDTH // 2, HEIGHT - 125))
            screen.blit(solved_text, text_rect)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
