from PySide6.QtWidgets import QApplication
from enum import Enum
import ConnectFour_GUI

class CurrentPlayer(Enum):
    HUMAN = 1
    COMPUTER = -1

if __name__ == '__main__':
    if QApplication.instance():
        app = QApplication.instance()
    else:
        app = QApplication([])
    player = CurrentPlayer.HUMAN
    gameMainWindow = ConnectFour_GUI.MainWindow()
    app.exec()
    def on_app_exit():
        app.quit()
        del app
    app.aboutToQuit.connect(on_app_exit)