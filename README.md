# Pac-Man AI Project

## Overview

This project involves implementing an AI agent for the classic Pac-Man game using the Minimax algorithm and an appropriate utility function. The goal is to enable the Pac-Man agent to play the game optimally by considering the positions of the ghosts and the available dots on the game board. The AI will be designed to maximize the score while avoiding capture by the ghosts.

### Minimax and Alpha-Beta Pruning

**Minimax Algorithm:**
Minimax is a decision-making algorithm used in two-player games. It evaluates all possible moves by recursively exploring game states up to a certain depth. The Maximizer aims to maximize the score, while the Minimizer seeks to minimize it. The algorithm chooses the move that maximizes the player's minimum gain, assuming the opponent plays optimally.

**Alpha-Beta Pruning:**
Alpha-Beta Pruning optimizes Minimax by eliminating branches that won't affect the final decision. It uses two values:
- **Alpha:** The best value that the Maximizer can achieve.
- **Beta:** The best value that the Minimizer can achieve.

If a node's score is outside the range defined by Alpha and Beta, further exploration of that node is skipped, improving efficiency without affecting the result.

## Key Features

- **Minimax Algorithm:** Used to determine the best move for Pac-Man by exploring potential future states up to a certain depth.
- **Expectimax Algorithm:** Applied to handle the randomness of ghost movements, improving the decision-making process by considering probabilistic outcomes.
- **Alpha-Beta Pruning:** Optional optimization to reduce the search space of the Minimax algorithm, allowing deeper search with improved efficiency.
- **Utility Function (e-utility):** Calculates the score of a game state based on Pac-Man's distance to ghosts and the nearest dot.
### Addressing a Key Challenge in Pac-Man AI

One of the major challenges in implementing AI for Pac-Man was dealing with the limitations of using Manhattan distance for finding the closest dot. The Manhattan distance approach often led to Pac-Man getting trapped behind walls, as it only considers the straight-line distance and does not account for obstacles in the path. This resulted in Pac-Man frequently being unable to navigate around barriers, making it more vulnerable to ghosts.

### Creative Solution Using BFS for Pathfinding

To address this issue, you applied a more sophisticated approach by utilizing the Breadth-First Search (BFS) algorithm. This method calculates the shortest path to the nearest dot while effectively navigating around obstacles like walls. Here’s a breakdown of how this solution overcomes the limitations of Manhattan distance and enhances Pac-Man’s performance:

1. **Challenge with Manhattan Distance:**
   - **Problem:** Manhattan distance only measures the horizontal and vertical distance between two points, ignoring any obstacles (walls) in between. This can mislead Pac-Man into making suboptimal decisions, such as getting stuck or taking a longer path around walls.
   - **Impact:** Pac-Man often ends up trapped, as the Manhattan distance does not provide a feasible path around obstacles, increasing the risk of encountering ghosts.

2. **BFS-Based Solution:**
   - **Solution:** By switching to BFS, you calculate the shortest possible path to the closest dot while accounting for walls and obstacles.
   - **Advantage:** BFS explores all possible paths level by level, ensuring that the first time it reaches a dot, it has found the shortest path to that dot. This method prevents Pac-Man from getting stuck behind walls, as it systematically checks all possible routes.

### Summary of Impact

By integrating BFS into the Pac-Man AI:

- **Obstacle Navigation:** Pac-Man can now navigate around walls effectively, avoiding getting stuck.
- **Enhanced Decision-Making:** The AI makes more informed decisions based on the shortest path rather than simple distance metrics, improving Pac-Man's survival and performance.
- **Reduced Risk:** The risk of encountering ghosts is minimized as Pac-Man avoids trapped paths and finds more efficient routes to the dots.

This creative solution showcases a significant advancement in the AI’s ability to handle complex environments, providing a more robust and intelligent gameplay experience.

## Classes

### PacMan
- **Attributes:**
  - `location`: Represents the current position of Pac-Man.
- **Methods:**
  - `update_location_by_move(move)`: Updates Pac-Man’s location based on the provided move (1: Up, 2: Right, 3: Down, 4: Left).

### Ghost
- **Attributes:**
  - `location`: Represents the current position of the ghost.
- **Methods:**
  - `update_location_by_move(move)`: Updates the ghost’s location based on the provided move.
  - `next_move()`: Determines the ghost’s next move randomly within the constraints of the maze.

### Game
- **Attributes:**
  - `maze`: A numpy array representing the game board.
  - `score`: Keeps track of Pac-Man’s score.
  - `goal_test_counter`: Tracks the number of remaining dots.
- **Methods:**
  - `dot_eaten(location)`: Updates the maze when a dot is eaten.
  - `game_print()`: Prints the current state of the game board.
  - `goal_test()`: Checks if the game has been won (all dots eaten).
  - `fail_test()`: Checks if Pac-Man has been caught by a ghost.
  - `possible_moves(location, maze)`: Determines the valid moves from a given location.
  - `e_utility(pacman, inky, pinky, maze)`: Calculates the utility score for a given game state.
  - `minimax(agent, depth, temp_pinky, temp_inky, temp_pacman, temp_maze, temp_GTC)`: Implements the Minimax algorithm to determine the best move.
  - `p_next_move()`: Selects Pac-Man’s next move using the Minimax algorithm.
  - `run()`: Executes the game loop, updating the game state and rendering the game board.

### Utility Functions
- **distance(location1, location2):** Computes the distance between two locations using BFS algorithm to avoid obstacles.
- **distance_M(location1, location2):** Calculates the Manhattan distance between two points.
- **clear():** Clears the console output for a smoother game simulation.

