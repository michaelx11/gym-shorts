import tic_tac_toe
import random_player
import random
import pickle
import argparse

class SarsaLookupAgent:

  ALPHA = .01
  DISCOUNT_FACTOR = .90
  MINIMUM_EPSILON = .01

  def __init__(self, modelString=None):
    # Action-value Q is simply a dictionary keyed on concatenated states
    self.qs = {}
    self.game = None
    self.step = 0
    if modelString:
      self.qs, self.step = pickle.loads(modelString)

  def getValidMovePairs(self, board):
    validMoves = []
    for r in range(3):
      for c in range(3):
        if board[r][c] == tic_tac_toe.GameEnvironment.EMPTY_STATE:
          validMoves.append((r, c))
    return validMoves

  # board state is [['X','X','O'], ..., ], action is (row, col)
  def getQKey(self, board, actionPair):
    return ''.join([''.join(row) for row in board] + map(str, actionPair))

  def getQValue(self, qKey):
    if not qKey in self.qs:
      self.qs[qKey] = 0.0
    return self.qs[qKey]

  # Choose action via epsilon-greedy approach
  def getAction(self):
    oldBoard = self.game.board
    self.step += 1
    validMoves = self.getValidMovePairs(oldBoard)
    randValue = random.random()
    if randValue < 1.0 / self.step or randValue < self.MINIMUM_EPSILON:
      # choose random action, default is 0,0 if no action possible (termination case)
      return random.choice(validMoves) if len(validMoves) > 0 else (0, 0)
    else:
      maxMoveValue = -10000000
      maxMove = (0, 0)
      for move in validMoves:
        qKey = self.getQKey(oldBoard, move)
        if self.getQValue(qKey) > maxMoveValue:
          maxMoveValue = self.getQValue(qKey)
          maxMove = move
      return maxMove

  def trainStep(self, oldState, oldAction, newState, newAction, reward):
    oldQKey = self.getQKey(oldState, oldAction)
    newQKey = self.getQKey(newState, newAction)
    oldQValue = self.getQValue(oldQKey)
    newQValue = self.getQValue(newQKey)
#    if oldQValue > 0 or reward != 0 or newQValue > 0:
#        print oldQValue, reward, newQValue
#        print oldState, oldAction
#        print newState, newAction

    self.qs[oldQKey] = oldQValue + self.ALPHA * (reward + self.DISCOUNT_FACTOR * newQValue - oldQValue)

  def runEpisode(self, shouldPrint=False, isTraining=False):
    self.game = tic_tac_toe.GameEnvironment()
    randomPlayer = random_player.RandomT3Player(self.game, 'O')
    board = self.game.board
    action = self.getAction()
    done = False
    reward = 0
    # while not done
    while not done:
      # Take action A, observe R and S'
      state = self.game.step(action[0], action[1])
      if shouldPrint:
        print 'After P1 move:', self.game.board
      newBoard, newAction, reward, done = state
      # Adversary moves (environment)
      if not done:
        randomMove = randomPlayer.chooseMove()
        adversaryStep = self.game.step(randomMove[0], randomMove[1])
      if shouldPrint:
        print 'After P2 move:', self.game.board
      # If the opponent wins, we need to replace the reward
      if not done and adversaryStep[3]:
        reward = adversaryStep[2]
        done = True
        state = adversaryStep
      # Recompute action, A', save old action
      oldAction = action
      action = self.getAction()
      # Now do the update equation
      if isTraining:
        self.trainStep(board, oldAction, self.game.board, action, reward)
      # save board state
      board = self.game.board
    if shouldPrint:
      print state
    return state

  def playGameWithUser(self):
    self.game = tic_tac_toe.GameEnvironment()
    board = self.game.board
    state = (self.game.board, (0,0), 0, False)
    action = self.getAction()
    # while not done
    while not state[3]:
      # Take action A, observe R and S'
      state = self.game.step(action[0], action[1])
      print 'After P1 move:', self.game.board
      # Adversary moves (environment)
      if not state[3]:
        print 'move? r,c:'
        randomMove = map(int, raw_input().split(','))
        state = adversaryStep = self.game.step(randomMove[0], randomMove[1])
      print 'After P2 move:', self.game.board
      # Recompute action, A', save old action
      action = self.getAction()
      # save board state
      board = self.game.board
    print state
    return state

  def getModelString(self):
    return pickle.dumps((self.qs, self.step))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--play', help="Start user interactive session against agent", action="store_true")
  parser.add_argument('--in_file', help="File to load model from", type=str)
  parser.add_argument('--out_file', help="File to write model to", type=str)
  args = parser.parse_args()

  if args.play:
    modelFileName = 'model.pickle'
    if args.in_file:
      modelFileName = args.in_file
    with open(modelFileName, 'r') as modelFile:
      modelString = modelFile.read()
      agent = SarsaLookupAgent(modelString=modelString)
      agent.playGameWithUser()
  else:
    agent = SarsaLookupAgent()
    if args.in_file:
      with open(args.in_file, 'r') as modelFile:
        modelString = modelFile.read()
        agent = SarsaLookupAgent(modelString=modelString)
    # Training
    for i in range(5000000):
      if i % 10000 == 0:
        print i
      agent.runEpisode(isTraining=True)

    # Evaluation
    numEvals = 10000
    numWins = 0
    numTies = 0
    numLosses = 0
    for i in range(numEvals):
      _, _, reward, _ = agent.runEpisode()
      if reward == -1:
        numLosses += 1
      elif reward == 0:
        numTies += 1
      elif reward == 1:
        numWins += 1
    print numWins, numTies, numLosses

    for i in range(5):
      print '\n\n\n\n-- Game: {} --\n\n'.format(i)
      agent.runEpisode(shouldPrint=True)

    outputFile = 'model.pickle'
    if args.out_file:
      outputFile = args.out_file
    with open(outputFile, 'w') as f:
      f.write(agent.getModelString())

