# This defines the game model / environment for tic-tac-toe
import operator

# 'X' is player one, also the agent
class GameEnvironment:

  EMPTY_STATE = ' '

  def __init__(self):
    self.board = [
        [self.EMPTY_STATE] * 3,
        [self.EMPTY_STATE] * 3,
        [self.EMPTY_STATE] * 3
    ]
    self.move = 'X'

  def getObservation(self):
    return self.board

  # returns True if provided player has won, False otherwise
  def playerHasWon(self, playerString):
    rows = [1] * 3
    cols = [1] * 3
    diags = [1] * 2
    for r in range(3):
      for c in range(3):
        isMatch = 1 if self.board[r][c] == playerString else 0
        rows[r] &= isMatch
        cols[c] &= isMatch
        if r == c:
          diags[0] *= isMatch
        if r + c == 2:
          diags[1] *= isMatch

    allChecks = rows + cols + diags
    return reduce(operator.__or__, allChecks)

  # returns 'X', 'O', or 'tie' if game finished otherwise false
  def checkBoard(self):
    gameFinished = True
    for r in self.board:
      for item in r:
        if item == self.EMPTY_STATE:
          gameFinished = False # game isn't finished
    if self.playerHasWon('X'):
      return 'X'
    elif self.playerHasWon('O'):
      return 'O'
    return 'TIE' if gameFinished else False

  # returns (new state, last action, reward, done)
  def step(self, row, col):
    # check that the square is not currently occupied
    if self.board[row][col] == self.EMPTY_STATE:
      self.board[row][col] = self.move
      self.move = 'X' if self.move != 'X' else 'O'
      winner = self.checkBoard()
      done = winner != False
      reward = 0
      if winner is 'X':
        reward = 1
      elif winner is 'O':
        reward = -1

      return (self.board, (row, col), reward, done)
    else:
      # Invalid move gets -1
      return (self.board, (row, col), -1, True)

if __name__ == '__main__':
  game = GameEnvironment()
  state = (game.board, (0, 0), 0, False)
  # while not done
  while not state[3]:
    print state[0]
    print 'enter row, col: '
    move = raw_input()
    movePair = map(int, move.split(','))
    print 'Making move {}'.format(movePair)
    state = game.step(movePair[0], movePair[1])
  print state
