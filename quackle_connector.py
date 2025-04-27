# scripts/quackle_connector.py
import subprocess
import time

class QuackleConnector:
    def __init__(self, quackle_path="./quacker"):
        self.process = subprocess.Popen(
            [quackle_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    def send_command(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_response(self, wait_time=0.1):
        time.sleep(wait_time)
        lines = []
        while True:
            line = self.process.stdout.readline()
            if not line.strip():
                break
            lines.append(line.strip())
        return lines

    def play_move(self, word, row, col, direction):
        # direction: 'H' or 'V'
        move_command = f"play {word} {row} {col} {direction}"
        self.send_command(move_command)

    def get_board(self):
        self.send_command("board")
        return self.read_response()

    def quit(self):
        self.send_command("quit")
        self.process.terminate()
