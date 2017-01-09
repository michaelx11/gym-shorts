import tic_tac_toe

import random

class RandomT3Player:

  def __init__(self, gameInstance, playerSymbol):
    self.game = gameInstance
    self.playerSymbol = playerSymbol

  def chooseMove(self):
    board = self.game.board
    validMoves = []
    for r in range(3):
      for c in range(3):
        if board[r][c] == tic_tac_toe.GameEnvironment.EMPTY_STATE:
          validMoves.append((r, c))

    return random.choice(validMoves)


if __name__ == '__main__':
  game = tic_tac_toe.GameEnvironment()
  randomPlayer = RandomT3Player(game, 'O')
  state = (game.board, (0, 0), 0, False)
  # while not done
  while not state[3]:
    print state[0]
    print 'enter row, col: '
    move = raw_input()
    movePair = map(int, move.split(','))
    print 'Making move {}'.format(movePair)
    state = game.step(movePair[0], movePair[1])

    randomMove = randomPlayer.chooseMove()
    print 'Random player chooses: {}'.format(randomMove)
    state = game.step(randomMove[0], randomMove[1])
