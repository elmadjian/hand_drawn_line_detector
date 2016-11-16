import numpy as np
import re

class Setup():
    def __init__(self, sysinput):
        self.seeds = []
        self.sinks = []
        self.colors = []
        textfile = self._get_text_file(sysinput)
        self.open_file(textfile)

    def open_file(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                data = line.split()
                if len(data) == 3:
                    self.seeds.append((int(data[1]), int(data[0])))
                elif len(data) == 2:
                    self.sinks.append((int(data[1]), int(data[0])))
        self._generate_colors(len(self.seeds))


    def _get_text_file(self, user_input):
        for i in user_input:
            if re.search("\w+.txt", i):
                return i


    def _generate_colors(self, size):
        for i in range(size):
            b = np.random.randint(50, 256)
            g = np.random.randint(50, 256)
            r = np.random.randint(50, 256)
            self.colors.append((b,g,r))
