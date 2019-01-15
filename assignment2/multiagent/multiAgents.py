# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        "initialize value according to current's score."
        initial_value = currentGameState.getScore()

        "avoid stopping"
        if action == Directions.STOP:
            return float("-inf")

        "find the min food distance,close to food is good."
        min_food_distance = 0
        for each_food in newFood.asList():
            food_distance = manhattanDistance(newPos,each_food)
            if min_food_distance == 0:
                min_food_distance = food_distance
            if food_distance < min_food_distance:
                min_food_distance = food_distance

        "find the min ghost distance,close to ghost is bad."
        min_ghost_distance = 0
        for each_ghost in newGhostStates:
            ghost_distance = manhattanDistance(newPos,each_ghost.getPosition())
            if min_ghost_distance == 0:
                min_ghost_distance = ghost_distance
            if ghost_distance < min_ghost_distance:
                min_ghost_distance = ghost_distance

        "Aovid Pacman hitting the ghost, the score should close to infinite small, becasue hitting ghost is a bad choice."
        if min_ghost_distance <= 1:
            return float("-inf")

        "if next position has a food on it, give it big reward."
        for food in currentGameState.getFood().asList():
            if manhattanDistance(newPos,food) == 0:
                return float("inf")

        """
        take 1/min ghost distance, because when ghost is close we want it has a big impact. When ghost is far away,
        we want it has a small impact on final value. (ghost far away is good, and close is bad.)
        """
        return initial_value-min_food_distance- 1/min_ghost_distance



def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        depth = self.depth
        result = self.maxPacman(gameState,depth)
        return result[0]


    def maxPacman(self,gameState,depth):

        "base case"
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return [None, self.evaluationFunction(gameState)]

        "For each action, call minGhost to find all the scores that min player return."
        new_game_state = []
        for action in gameState.getLegalActions(0):
            new_game_state.append([action,(self.minGhost(gameState.generateSuccessor(0,action),depth,1))[1]])

        "Pick the highest value among all the value that ghost return."
        max_result = []
        for state in new_game_state:
            if max_result == []:
                max_result = state
            if state[1] > max_result[1]:
                max_result = state

        return max_result


    def minGhost(self,gameState,depth,ghostIndex):

        "base case"
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return [None, self.evaluationFunction(gameState)]


        new_game_state = []
        for action in gameState.getLegalActions(ghostIndex):
            "if it is not the last ghost,recursive call."
            if ghostIndex != (gameState.getNumAgents() - 1):
                new_game_state.append([action, (self.minGhost(gameState.generateSuccessor(ghostIndex,action),depth,ghostIndex + 1))[1]])
            "after we go through all the ghost. It is max's turn to play."
            if ghostIndex == (gameState.getNumAgents()-1):
                new_game_state.append([action,(self.maxPacman(gameState.generateSuccessor(ghostIndex,action),depth - 1))[1]])

        "find the smallest value since ghost is min player."
        min_result = []
        for state in new_game_state:
            if min_result == []:
                min_result = state
            if state[1] < min_result[1]:
                min_result = state

        return min_result


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        alpha = float("-inf")
        beta = float("inf")
        result = self.maxPacman(gameState,depth,alpha,beta)
        return result[0]

    def maxPacman(self,gameState,depth,alpha, beta):

        "base case"
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return [None, self.evaluationFunction(gameState)]
        new_game_state = []
        max_result = []

        "For each action, call minGhost to find the scores that min player return."
        for action in gameState.getLegalActions(0):
            new_game_state.append([action,(self.minGhost(gameState.generateSuccessor(0,action),depth,1,alpha, beta))[1]])

            for state in new_game_state:
                if max_result == []:
                    max_result = state
                if state[1] > max_result[1]:
                    max_result = state

            "if current max node's value is greater or equal to min node's beta,return."
            if max_result[1] >= beta:
                return max_result
            "otherwise, update alpha."
            alpha = max(alpha,max_result[1])

        return max_result


    def minGhost(self,gameState,depth,ghostIndex,alpha, beta):

        "base case"
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return [None, self.evaluationFunction(gameState)]

        new_game_state = []
        min_result = []

        for action in gameState.getLegalActions(ghostIndex):
            "if it is not the last ghost,recursive call."
            if ghostIndex != (gameState.getNumAgents() - 1):
                new_game_state.append([action, (self.minGhost(gameState.generateSuccessor(ghostIndex,action),depth,ghostIndex + 1,alpha,beta))[1]])
            "after we go through all the ghost. It is max's turn to play."
            if ghostIndex == (gameState.getNumAgents()-1):
                new_game_state.append([action,(self.maxPacman(gameState.generateSuccessor(ghostIndex,action),depth - 1,alpha, beta))[1]])

            for state in new_game_state:
                if min_result == []:
                    min_result = state
                if state[1] < min_result[1]:
                    min_result = state
            "if current min node's value is less or equal to alpha, return. "
            if min_result[1] <= alpha:
                return min_result
            "otherwise, update beta."
            beta = min(beta,min_result[1])

        return min_result





class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        result = self.maxplayer(gameState, depth)
        return result[0]

    def maxplayer(self, gameState, depth):

        "base case"
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return [None, self.evaluationFunction(gameState)]
        new_game_state = []

        "For each action, call minGhost to find all the average that min player return."
        for action in gameState.getLegalActions(0):
            new_game_state.append([action, (self.chanceplayer(gameState.generateSuccessor(0, action), depth, 1))])

        "Pick the highest value among all the value that ghost return."
        max_result = []
        for state in new_game_state:
            if max_result == []:
                max_result = state
            if state[1] > max_result[1]:
                max_result = state

        return max_result

    def chanceplayer(self, gameState, depth, ghostIndex):

        "base case"
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        new_game_state = []

        for action in gameState.getLegalActions(ghostIndex):
            "if it is not the last ghost,recursive call."
            if ghostIndex != (gameState.getNumAgents() - 1):
                new_game_state.append((self.chanceplayer(gameState.generateSuccessor(ghostIndex, action), depth, ghostIndex + 1)))
            "after we go through all the ghost. It is max's turn to play."
            if ghostIndex == (gameState.getNumAgents() - 1):
                new_game_state.append((self.maxplayer(gameState.generateSuccessor(ghostIndex, action), depth - 1))[1])


        "find the average since we are choosing uniformly at random."
        sum_of_state = 0
        for state in new_game_state:
            sum_of_state += state

        return float(sum_of_state)/float(len(new_game_state))



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood()
    curghost = currentGameState.getGhostStates()
    curCapsules = currentGameState.getCapsules()

    initial_value = currentGameState.getScore()


    "find the min food distance"
    min_food_distance = 0
    for each_food in curFood.asList():
        food_distance = manhattanDistance(curPos, each_food)
        if min_food_distance == 0:
            min_food_distance = food_distance
        if food_distance < min_food_distance:
            min_food_distance = food_distance


    "find the min capsules distance,take caspsules into consideration."
    min_capsules_distance = 0
    for each_capsule in curCapsules:
        capsule_distance = manhattanDistance(curPos, each_capsule)
        if min_capsules_distance == 0:
            min_capsules_distance = capsule_distance
        if capsule_distance < min_capsules_distance:
            min_capsules_distance = capsule_distance


    "find the min ghost distance"
    min_ghost_distance = 0
    for each_ghost in curghost:
        ghost_distance = manhattanDistance(curPos, each_ghost.getPosition())
        if min_ghost_distance == 0:
            min_ghost_distance = ghost_distance
        if ghost_distance < min_ghost_distance:
            min_ghost_distance = ghost_distance

    "Aovid Pacman hitting the ghost, the score should close to infinite small, becasue hitting ghost is a bad choice."
    if min_ghost_distance <= 1:
        return float("-inf")

    "if next position has a food on it, give it big reward."
    for food in currentGameState.getFood().asList():
        if manhattanDistance(curPos, food) == 0:
            return float("inf")

    return initial_value-min_food_distance-min_capsules_distance- 1/min_ghost_distance


# Abbreviation
better = betterEvaluationFunction

