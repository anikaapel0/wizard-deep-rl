from tkinter import *


class WizardWindow(object):

    def __init__(self):
        self.root = None
        self._players_header = []
        self._players_score = []
        self._players_predictions = []
        self._players_tricks = []
        self._trick_cars = []
        self._handcards = []
        self._init_root()

    def _init_root(self):
        self.root = Tk()

        self.root.title("Wizard")
        self.root.geometry("1000x1000")

        # Label für vier Spieler hinzufügen
        self.add_new_player("Player 1", x=100, y=0)
        self.add_new_player("Player 2", x=900, y=100)
        self.add_new_player("Player 3", x=100, y=900)
        self.add_new_player("Player 4", x=0, y=100)

        # label anzeigen, auf Eingabe des Benutzers warten
        self.root.mainloop()

    def add_new_player(self, text, x, y):
        add_y = 20
        player = Label(self.root, text=text)
        player.place(x=x, y=y)
        self._players_header.append(player)

        y += add_y
        player_score = Label(self.root, text="Score: xx")
        player_score.place(x=x, y=y)
        self._players_score.append(player_score)

        y += add_y
        player_predicted = Label(self.root, text="Predicted: xx")
        player_predicted.place(x=x, y=y)
        self._players_predictions.append(player_predicted)

        y += add_y
        player_won = Label(self.root, text="Tricks Won: xx")
        player_won.place(x=x, y=y)
        self._players_tricks.append(player_won)


if __name__ == "__main__":
    window = WizardWindow()
