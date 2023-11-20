from cmath import inf
import queue
import numpy as np
import random as rd
from os import system, name
from time import sleep
import math
import copy
from collections import deque
import networkx as nx


#defining the maze and its obstacles and dots with 0 : no obstacles or dots, 1 : dots, -1 : obstacle
maze_initial_map = np.array([[1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1],
         [1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1],
         [1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1],
         [1, -1, 1, -1, -1, 1, -1, -1, 0, 0, -1, -1, 1, -1, -1, 1, -1, 1],
         [1, 1, 1, 1, 1, 1, -1, 0, 0, 0, 0, -1, 1, 1, 1, 1, 1, 1],
         [1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1],
         [1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1],
         [1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1],
         [1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1]])
         

#clear
def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')

#the function used for calculating distance
def distance(pacman_location, maze):
    visited = []
    next_to_visit = []
    distance_nodes = {tuple(pacman_location): 0}

    next_to_visit.append(pacman_location)
    visited.append(pacman_location)

    while len(next_to_visit):
        current_node = next_to_visit.pop(0)

        if maze[current_node[0]][current_node[1]] == 1:
            return distance_nodes[tuple(current_node)]
        
        if (current_node[0] - 1 >= 0) and (maze[current_node[0] - 1][current_node[1]] != -1):
            if ([current_node[0] - 1, current_node[1]] not in visited):
                next_to_visit.append([current_node[0] - 1, current_node[1]])
                visited.append([current_node[0] - 1, current_node[1]])
                distance_nodes[tuple([current_node[0] - 1, current_node[1]])] = distance_nodes[tuple(current_node)] + 1

        if (current_node[0] + 1 <= 8) and (maze[current_node[0] + 1][current_node[1]] != -1):
            if ([current_node[0] + 1, current_node[1]] not in visited):
                next_to_visit.append([current_node[0] + 1, current_node[1]])
                visited.append([current_node[0] + 1, current_node[1]])
                distance_nodes[tuple([current_node[0] + 1, current_node[1]])] = distance_nodes[tuple(current_node)] + 1


        if (current_node[1] - 1 >= 0) and (maze[current_node[0]][current_node[1] - 1] != -1):
            if ([current_node[0], current_node[1] - 1] not in visited):
                next_to_visit.append([current_node[0], current_node[1] - 1])
                visited.append([current_node[0], current_node[1] - 1])
                distance_nodes[tuple([current_node[0], current_node[1] - 1])] = distance_nodes[tuple(current_node)] + 1

        if (current_node[1] + 1 <= 17) and (maze[current_node[0]][current_node[1] + 1] != -1):
            if ([current_node[0], current_node[1] + 1] not in visited):
                next_to_visit.append([current_node[0], current_node[1] + 1])
                visited.append([current_node[0], current_node[1] + 1])
                distance_nodes[tuple([current_node[0], current_node[1] + 1])] = distance_nodes[tuple(current_node)] + 1

    return 0

#the function used for manhatan distance
def distance_M(location1, location2):
    result = abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])
    return result


#defining the pacman class
class PacMan:
    def __init__(self, location):
        self._location = location

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, new_location):
        self._location = new_location

#1 : up, 2 : right, 3 : down, 4 : left
    def update_location_by_move(self, move):
        if move == 1:
            self._location[0] = self._location[0] - 1
        
        if move == 2:
            self._location[1] = self._location[1] + 1
        
        if move == 3:
            self._location[0] = self._location[0] + 1
        
        if move == 4:
            self._location[1] = self._location[1] - 1


#defining the ghost class
class Ghost:
    def __init__(self, location):
        self._location = location

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, new_location):
        self._location = new_location


    #1 : up, 2 : right, 3 : down, 4 : left
    def update_location_by_move(self, move):
        if move == 1:
            self._location[0] = self._location[0] - 1
        
        if move == 2:
            self._location[1] = self._location[1] + 1
        
        if move == 3:
            self._location[0] = self._location[0] + 1
        
        if move == 4:
            self._location[1] = self._location[1] - 1

    #finding the next move for the ghost randomly & not going into the obsatcles
    #1 : up, 2 : right, 3 : down, 4 : left
    def next_move(self, maze):
        random_number = rd.randint(1, 4)
# temp_location = self._location

        if random_number == 1:
#temp_location[0] = temp_location[0] - 1
            if (self._location[0] - 1 >= 0) and (maze[self._location[0] - 1][self._location[1]] != -1):
                self._location[0] = self._location[0] - 1

        if random_number == 2:
        #   temp_location[1] = temp_location[1] + 1
            if (self._location[1] + 1 <= 17) and (maze[self._location[0]][self._location[1] + 1] != -1):
                self._location[1] = self._location[1] + 1

        if random_number == 3:
        #   temp_location[0] = temp_location[0] + 1
            if (self._location[0] + 1 <= 8) and (maze[self._location[0] + 1][self._location[1]] != -1):
                self._location[0] = self._location[0] + 1

        if random_number == 4:
        #   temp_location[1] = temp_location[1] - 1
            if (self._location[1] - 1 >= 0) and (maze[self._location[0]][self._location[1] - 1] != -1):
                self._location[1] = self._location[1] - 1


#defining the game class
class Game:
    def __init__(self):
        self._maze = np.copy(maze_initial_map)
    #getting one instance from the class, PacMan
        self._pacman = PacMan([8, 0])
    #getting two instances from the class, Ghost, naming them inky and pinky
        self._inky = Ghost([4, 8])
        self._pinky = Ghost([4, 9])
    #score of the game
        self._score = 0
    #the number of the remaining dots
        self._goal_test_counter = 100

    @property
    def maze(self):
        return self._maze
        
    @maze.setter
    def maze(self, new_maze):
        self._maze = np.copy(new_maze)

    @property
    def pacman(self):
        return self._pacman

    @pacman.setter
    def pacman(self, new_pacman):
        self._pacman = new_pacman

    @property
    def inky(self):
        return self._inky

    @inky.setter
    def inky(self, new_inky):
        self._inky = new_inky

    @property
    def pinky(self):
        return self._pinky

    @pinky.setter
    def pinky(self, new_pinky):
        self._pinky = new_pinky

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, new_score):
        self._score = new_score

    @property
    def goal_test_counter(self):
        return self._goal_test_counter

    @goal_test_counter.setter
    def goal_test_counter(self, new_GTC):
        self._goal_test_counter = new_GTC

    #the function for updating the map when a move is done
    def dot_eaten(self, coordinates):
        self._maze[coordinates[0]][coordinates[1]] = 0

    #the function for printing the maze
    def game_print(self):

        print("Score : ", self._score)

        for i in range (np.shape(self._maze)[0]):
            for j in range (np.shape(self._maze)[1]):

                if self._inky._location == [i, j]:
                    print('I', end=" ")
                    continue

                if self._pinky._location == [i, j]:
                    print('P', end=" ")
                    continue

                if self._pacman._location == [i, j]:
                    print('M', end=" ")
                    continue

                if self._maze[i][j] == 1:
                    print('.', end=" ")


                if self._maze[i][j] == 0:
                    print(' ', end=" ")

                if self._maze[i][j] == -1:
                    print('@', end=" ")
            print()

    #defining the goal test of the game which is a function that tests whether all of the dots are eaten or not
    def goal_test(self):
        if self._goal_test_counter == 0:
            return 0
        else:
            return 1

    #defining the fail test of the game which is a function that tests whether the pacman and either one of the ghosts is in the same coordinates or not
    def fail_test(self):
        if self._pacman._location == self._inky._location or self._pacman._location == self._pinky._location:
            return 0
        else:
            return 1

    #1 : up, 2 : right, 3 : down, 4 : left
    #defining the possible moves function that returns the moves you can have from a curtain location
    def possible_moves(self, location, maze):
        moves = []
        for i in range (1, 5):
            if i == 1:
                if (location[0] - 1 >= 0) and (maze[location[0] - 1][location[1]] != -1):
                    moves.append(1)
                else:
                    continue

            if i == 2:
                if (location[1] + 1 <= 17) and (maze[location[0]][location[1] + 1] != -1):
                    moves.append(2)
                else:
                    continue

            if i == 3:
                if (location[0] + 1 <= 8) and (maze[location[0] + 1][location[1]] != -1):
                    moves.append(3)
                else:
                    continue

            if i == 4:
                if (location[1] - 1 >= 0) and (maze[location[0]][location[1] - 1] != -1):
                    moves.append(4)
                else:
                    continue
        return moves

    #the e-utility function that is for calculating the score for each state
    def e_utility(self, maze, pacman, inky, pinky):
        initial_utility = 0
        radius = 3
        PI = distance_M(pacman._location, inky._location)
        if PI <= radius:
            if PI == 3:
                initial_utility -= 2
            if PI == 2:
                initial_utility -= 4
            if PI == 1:
                initial_utility -= 6
            if PI == 0:
                initial_utility -= 200

        PP = distance_M(pacman._location, pinky._location)
        if PP <= radius:
            if PP == 3:
                initial_utility -= 2
            if PP == 2:
                initial_utility -= 4
            if PP == 1:
                initial_utility -= 6
            if PP == 0:
                initial_utility -= 200


        CD = distance(pacman._location, maze)
        initial_utility += 3 * (100 - CD)


        return initial_utility








    # the minimax algorithm, used for pacman's next move
    def minimax(self, agent, depth, temp_maze, temp_pacman, temp_inky, temp_pinky, temp_GTC):
        if (temp_GTC == 0) or (temp_pacman._location == temp_inky._location) or (temp_pacman._location == temp_pinky._location) or (depth == 0):
            return self.e_utility(temp_maze, temp_pacman, temp_inky, temp_pinky)
        
        #pacman : 0
        if agent == 0:
            u = 0
            flag = 0
            max_value = float('-inf')
            for moves in self.possible_moves(temp_pacman._location, temp_maze):
                temp_pacman.update_location_by_move(moves)
                if temp_maze[temp_pacman._location[0]][temp_pacman._location[1]] == 1:
                    temp_GTC -= 1
                    u += 4
                    temp_maze[temp_pacman._location[0]][temp_pacman._location[1]] = 0
                    # flag = 1

                new_value = self.minimax((agent + 1) % 3, depth - 1, temp_maze, temp_pacman, temp_inky, temp_pinky, temp_GTC) + 10
                max_value = max(new_value, max_value)

            return max_value
        
        #inky : 1
        elif agent == 1:
            min_value = float('inf')
            for moves in self.possible_moves(temp_inky._location, temp_maze):
                temp_inky.update_location_by_move(moves)
                new_value = self.minimax((agent + 1) % 3, depth - 1, temp_maze, temp_pacman, temp_inky, temp_pinky, temp_GTC)
                min_value = min(new_value, min_value)

            return min_value
        
        #pinky : 2
        elif agent == 2:
            min_value = float('inf')
            for moves in self.possible_moves(temp_pinky._location, temp_maze):
                temp_pinky.update_location_by_move(moves)
                new_value = self.minimax((agent + 1) % 3, depth - 1, temp_maze, temp_pacman, temp_inky, temp_pinky, temp_GTC)
                min_value = min(new_value, min_value)

            return min_value

    #the function for the next move of the pacman
    #usingthe minimax algorithm for choosing the successive move for the pacman & updating the goat_test_counter if necessary & updating the maze & not going into the obstacles & showing the pacman's score until now
    def P_next_move(self):
        self._score -= 1
        temp_pacman = copy.deepcopy(self._pacman)
        temp_inky = copy.deepcopy(self._inky)
        temp_pinky = copy.deepcopy(self._pinky)
        temp_maze = copy.deepcopy(self._maze)
        temp_GTC = copy.deepcopy(self._goal_test_counter)
        initial_utility = 0
        best_utility = 0

        for move in self.possible_moves(temp_pacman._location, temp_maze):

            temp_pacman.update_location_by_move(move)

            if temp_maze[temp_pacman._location[0]][temp_pacman._location[1]] == 1:
                    temp_GTC -= 1
                    initial_utility += 4

                    temp_maze[temp_pacman._location[0]][temp_pacman._location[1]] = 0



            utility = self.minimax(1, 3, temp_maze, temp_pacman, temp_inky, temp_pinky, temp_GTC) + initial_utility
            if utility > best_utility:
                best_utility = utility
                best_move = move
        
        self._pacman.update_location_by_move(best_move)
        if self._maze[self._pacman._location[0]][self._pacman._location[1]] == 1:
                    self._goal_test_counter -= 1
                    self._score += 10
                    self.dot_eaten([self._pacman._location[0], self._pacman._location[1]])

    #the game function
    def run(self):
        self._score += 10
        self._maze[8][0] = 0
        self._goal_test_counter -= 1
        self.game_print()
        sleep(0.5)
        clear()
        while(self.goal_test() or self.fail_test()):
            self.P_next_move()
            self._inky.next_move(self._maze)
            self._pinky.next_move(self._maze)
            self.game_print()
            sleep(0.5)
            clear()
        if self.goal_test():
            print("Pacman won!")
        if self.fail_test():
            print("Pacman lost!")


game = Game()
game.run()
