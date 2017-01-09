import tic_tac_toe
import random_player
import random
import pickle

class SarsaLookupAgent:

  LEARNING_FACTOR = .99
  DISCOUNT_FACTOR = .99

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
    random.shuffle(validMoves)
    if random.random() < 1.0 / self.step:
      # choose random action
      return random.choice(validMoves)
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

    self.qs[oldQKey] = oldQValue + self.LEARNING_FACTOR * (reward + self.DISCOUNT_FACTOR * newQValue - oldQValue)

  def runEpisode(self, shouldPrint=False, isTraining=False):
    self.game = tic_tac_toe.GameEnvironment()
    randomPlayer = random_player.RandomT3Player(self.game, 'O')
    state = (self.game.board, (0, 0), 0, False)
    # while not done
    while not state[3]:
      oldBoard, oldAction = state[:2]
      movePair = self.getAction()
      state = self.game.step(movePair[0], movePair[1])
      newBoard, newAction, reward, done = state
      if shouldPrint:
        print 'After P1 move:', self.game.board
      if isTraining:
        self.trainStep(oldBoard, oldAction, newBoard, newAction, reward)
      if done:
        break
      randomMove = randomPlayer.chooseMove()
      state = self.game.step(randomMove[0], randomMove[1])
      if shouldPrint:
        print 'After P2 move:', self.game.board
    if shouldPrint:
      print state
    return state

  def getModelString(self):
    return pickle.dumps((self.qs, self.step))

if __name__ == '__main__':
  agent = SarsaLookupAgent()
  # Training
  for i in range(10000):
    agent.runEpisode(isTraining=True)

  # Evaluation
  numEvals = 1000
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

  with open('model.pickle', 'w') as f:
    f.write(agent.getModelString())

