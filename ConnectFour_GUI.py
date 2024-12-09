import AI_agent
import ConnectFour_Logic
import asyncio
from enum import Enum
import functools
import numpy as np
import torch
from PySide6.QtCore import QTimer, Qt, Signal, QRect, QThread
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QBrush, QCursor
from PySide6.QtWidgets import QWidget, QPushButton, QMainWindow, QSizePolicy, QTextBrowser
import numpy as np
import time
import random

class CurrentPlayer(Enum):
    HUMAN = 1
    COMPUTER = -1

class gameStatus(Enum):
    HUMAN_LOST = -1
    DRAW = 0
    HUMAN_WON = 1
    IN_PROGRESS = 2
    
class AlphaZero(QThread):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent().computerMoveSignal.connect(self.parent().board.computerMove)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game = ConnectFour_Logic.ConnectFour()
        self.args = {
            'C': 2,
            'num_searches': 100,
            'dirichlet_epsilon': 0.0,
            'dirichlet_alpha': 0.3
        }
        self.model = AI_agent.ResNet(self.game, 9, 128, self.device)
        self.model.load_state_dict(torch.load("./model_7_ConnectFour.pt", map_location=self.device))
        self.model.eval()
        self.mcts = AI_agent.MCTS(self.game, self.args, self.model)

    def computeAIMove(self, moves, value):
        neutral_state = self.game.change_perspective(moves, value)
        mcts_probs = self.mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
        if moves[1, action] != 0:
            self.parent().board.button_i_status[action] = False
        moves = self.game.get_next_state(moves, action, value)
        val, is_terminal = self.game.get_value_and_terminated(moves, action)
        return_dict = {"moves": moves, "val": val, "is_terminal": is_terminal}
        return return_dict
    
    def run(self):
        if self.parent().status == gameStatus.IN_PROGRESS and self.parent().player == CurrentPlayer.COMPUTER:
            result = self.computeAIMove(
                self.parent().board.moves,
                self.parent().player.value
                )
            self.parent().computerMoveSignal.emit(result)
        self.quit()

    def startWorker(self):
        self.start()

class gameInfoWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.messages = [
            """
            <html>
            <body style="background-color: #000000;">
            <div style="text-align: center;">
                <p><span style="font-size: 72px;">🙁</span></p>
                <p style="color:#00ff00;"><h1>You Lost!</h1></p>
            </div>
            </body>
            </html>
            """,
            """
            <html>
            <body style="background-color: #000000;">
            <div style="text-align: center;">
                <p><span style="font-size: 72px;">😅</span></p>
                <p style="color:#ffffff;"><h1>phew... Draw!</h1></p>
            </div>
            </body>
            </html>
            """,
            """
            <html>
            <body style="background-color: #000000;">
            <div style="text-align: center;">
                <p><span style="font-size: 72px;">🥳</span></p>
                <p style="color:#0000ff;"><h1>You Won!</h1></p>
            </div>
            </body>
            </html>
            """,
            """
            <html>
            <body style="background-color: #000000;">
            <div style="text-align: center;">
                <p><span style="font-size: 72px;">⌛</span></p>
                <p style="color:#ffffff;"><h1>Game In Progress</h1></p>
            </div>
            </body>
            </html>
            """
        ]

        self.text_browser = QTextBrowser(self)
        self.text_browser.setOpenExternalLinks(True)  # Enable links if needed
        self.text_browser.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_browser.setStyleSheet("border: none;")
        self.show()

    def paintEvent(self, event):
        super().paintEvent(event)
        self.text_browser.setHtml(self.messages[self.parent().status.value + 1])
        
class Timer(QThread):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent().buttonClickedSignal.connect(self.resetTimer)
        self.parent().timerResetSignal.connect(self.resetTimer)
        self.parent().gameOverSignal.connect(self.resetTimer)
        self.timer = QTimer(self)
        self.time_elapsed = 0
        self.max_time_ms = 30 * 1000
        self.parent().timeElapsedSignal.connect(self.parent().timerGUI.timeElapsed)
        self.timer.timeout.connect(self.updateTimer)
        self.timer.start(20)

    def updateTimer(self):
        self.time_elapsed += 20
        if self.time_elapsed >= self.max_time_ms:
            self.timer.stop()
            self.parent().status = gameStatus.HUMAN_LOST
            self.parent().gameOverSignal.emit()
            self.timer.quit()
            self.quit()
        self.parent().timeElapsedSignal.emit(self.time_elapsed)
    
    def resetTimer(self):
        self.time_elapsed = 0
        if self.parent().status != gameStatus.IN_PROGRESS:
            self.timer.stop()
            self.quit()
        self.parent().timeElapsedSignal.emit(self.time_elapsed)

class TimerWidget(QWidget):
    def __init__(self, parent, cellWidth):
        super().__init__(parent)
        self.cellWidth = cellWidth
        self.max_time_ms = 30 * 1000
        self.time_elapsed = 0

    def timeElapsed(self, time):
        self.time_elapsed = time
        self.repaint()

    def paintEvent(self, event):
        qp = QPainter(self)
        if not qp.isActive():
            qp.begin(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.fillRect(self.rect(), QBrush(QColor(0, 0, 0)))
        rect = self.rect()
        w = min(rect.width(), rect.height()) - self.cellWidth
        rect = QRect(self.cellWidth/2, self.cellWidth/2, w, w)
        start_angle = 90 * 16 
        span_angle = abs(int((self.max_time_ms - self.time_elapsed) / self.max_time_ms * 360 * 16))
        pen = QPen()
        pen.setColor(QColor(255, 255, 255))
        pen.setWidth(10)
        qp.setPen(pen)
        qp.drawEllipse(rect)
        if self.parent().player == CurrentPlayer.COMPUTER:
            pen.setColor(QColor(0, 255, 0))
        else:
            pen.setColor(QColor(0, 0, 255))
        pen.setWidth(11)
        qp.setPen(pen)
        qp.drawArc(rect, start_angle, span_angle)
        font = QFont()
        font.setPointSize(14)
        qp.setFont(font)
        time_left_in_seconds = int(abs(self.max_time_ms - self.time_elapsed)/1000)
        qp.drawText(rect, Qt.AlignCenter, f"Time Left:\n{time_left_in_seconds} seconds")
        qp.end()

class Connect4Board(QWidget):
    def __init__(self, parent, cellWidth):
        super().__init__(parent)
        self.cellWidth = cellWidth
        self.c4logic = ConnectFour_Logic.ConnectFour()
        self.buttons = []
        self.button_i_status = [True for _ in range(7)]
        self.moves = np.array([[0 for _ in range(7)] for _ in range(6)])
        self.parent().gameOverSignal.connect(self.disableButtons)
        self.initUI()

    def initUI(self):
        self.setGeometry(self.rect())
        self.createButtons()
        self.show()

    def createButtons(self):
        font = QFont("Arial", 15) 
        for column in range(7):
            self.buttons.append(QPushButton(f"{column+1}", self))
            self.buttons[column].setFont(font)
            self.buttons[column].setGeometry((column+0.75)*self.cellWidth, 6.75*self.cellWidth, self.cellWidth/2, self.cellWidth/2)
            self.buttons[column].clicked.connect(functools.partial(self.buttonClicked, column))
        if self.parent().player == CurrentPlayer.COMPUTER:
            self.disableButtons()

    def disableButtons(self):
        for column in range(0, 7):
            self.buttons[column].setEnabled(False)

    def paintEvent(self, event):
        qp = QPainter(self)
        if not qp.isActive():
            qp.begin(self)
        qp.setRenderHint(QPainter.Antialiasing)
        self.drawBackground(qp)
        self.drawGrid(qp)
        self.drawMoves(qp)
        qp.end()

    def drawBackground(self, qp):
        qp.fillRect(self.rect(), QBrush(QColor(0, 0, 0)))
        
    def drawGrid(self, qp):
        qp.setPen(Qt.white)
        qp.setBrush(Qt.white)
        for i in range(0, 7):
           qp.drawLine(self.cellWidth/2, (i+0.5)*self.cellWidth, 7.5*self.cellWidth, (i+0.5)*self.cellWidth)
        for i in range(0, 8):
           qp.drawLine((i+0.5)*self.cellWidth, self.cellWidth/2, (i+0.5)*self.cellWidth, 6.5*self.cellWidth)

    def drawMoves(self, qp):
        for i in range(0, 6):
            for j in range(0, 7):
                if self.moves[i][j]:
                    if self.moves[i][j] == CurrentPlayer.COMPUTER.value:
                        qp.setPen(QColor(0, 255, 0))
                        qp.setBrush(QColor(0, 255, 0))
                    else:
                        qp.setPen(QColor(0, 0, 255))
                        qp.setBrush(QColor(0, 0, 255))
                    qp.drawEllipse((j+0.6)*self.cellWidth, (i+0.6)*self.cellWidth, 0.8*self.cellWidth, 0.8*self.cellWidth)
            
    def buttonClicked(self, column):
        row = np.max(np.where(self.moves[:, column] == 0))
        if row == 0:
            self.buttons[column].setEnabled(False)
        self.moves[row][column] = self.parent().player.value
        self.parent().repaint()
        for col in range(7):
            self.buttons[col].setEnabled(False)
        if self.c4logic.check_win(self.moves, column):
            self.parent().gameOverSignal.emit()
            self.parent().status = gameStatus.HUMAN_WON
        self.parent().buttonClickedSignal.emit()
        self.parent().alphazero.startWorker()
        self.parent().switchPlayer()

    def computerMove(self, data):
        self.moves = data["moves"]
        val = data["val"]
        is_terminal = data["is_terminal"]
        if is_terminal:
            print(self.moves)
            if val == 1:
                print(self.parent().player, "won")
                if self.parent().player.value == 1:
                    self.parent().status = gameStatus.HUMAN_WON
                elif self.parent().player.value == -1:
                    self.parent().status = gameStatus.HUMAN_LOST
            else:
                self.parent().status = gameStatus.DRAW
            self.parent().gameOverSignal.emit()
            self.parent().update()
        else:
            self.parent().switchPlayer()
            self.parent().timerResetSignal.emit()
            for column in range(7):
                if self.button_i_status[column]:
                    self.buttons[column].setEnabled(True)
            self.parent().repaint()

class mainWindow(QMainWindow):
    buttonClickedSignal = Signal()
    timerResetSignal = Signal()
    gameOverSignal = Signal()
    computerMoveSignal = Signal(dict)
    timeElapsedSignal = Signal(int)

    
    def __init__(self):
        super().__init__()
        self.player = CurrentPlayer(self.choosePlayer())
        self.status = gameStatus.IN_PROGRESS
        self.cellWidth = 80
        self.playerAction = 0
        self.board = Connect4Board(self, self.cellWidth)
        self.alphazero = AlphaZero(self)
        self.alphazero.start()
        self.timerGUI = TimerWidget(self, self.cellWidth)
        self.timer = Timer(self)
        self.timer.start()
        self.gameInfo = gameInfoWidget(self)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Connect-Four")
        self.setGeometry(0, 0, 10.5*self.cellWidth, 7.5*self.cellWidth)

        self.board.setGeometry(0, 0, 7.5*self.cellWidth, 7.5*self.cellWidth)
        self.timerGUI.setGeometry(7.5*self.cellWidth, 0, 3*self.cellWidth, 3*self.cellWidth)
        self.gameInfo.setGeometry(7.5*self.cellWidth, 3.5*self.cellWidth, 3*self.cellWidth, 5*self.cellWidth)

        self.board.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.timerGUI.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.gameInfo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.board.setMinimumSize(7.5*self.cellWidth, 7.5*self.cellWidth)
        self.timerGUI.setMinimumSize(3*self.cellWidth, 3*self.cellWidth)
        self.gameInfo.setMinimumSize(3*self.cellWidth, 5*self.cellWidth)

        self.show()

    def paintEvent(self, event):
        qp = QPainter(self)
        if not qp.isActive():
            qp.begin(self)
        qp.fillRect(self.rect(), QBrush(QColor(0, 0, 0)))
        qp.end()

    def choosePlayer(self):
        random.seed(time.time())
        r = random.choice([1,-1])
        return r

    def switchPlayer(self):
        if self.player == CurrentPlayer.COMPUTER:
            self.player = CurrentPlayer.HUMAN
        else:
            self.player = CurrentPlayer.COMPUTER
