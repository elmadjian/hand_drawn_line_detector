import numpy as np
import re

class Setup():
    def __init__(self, sysinput):
        self.seeds = []
        self.sinks = []
        self.colors = []
        self.correct = {}
        textfile = self._get_text_file(sysinput)
        self.open_file(textfile)


    def open_file(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                data = line.split()
                if len(data) == 3:
                    seed = (int(data[1]), int(data[0]))
                    self.seeds.append(seed)
                    self.correct[seed] = int(data[2])
                elif len(data) == 2:
                    self.sinks.append((int(data[1]), int(data[0])))
        self._generate_colors(len(self.seeds))
        self._transform_correct()


    def _get_text_file(self, user_input):
        for i in user_input:
            if re.search("\w+.txt", i):
                return i
            if re.search("\w+.[a-z]+", i):
                return i[:-3] + 'txt'


    def _generate_colors(self, size):
        for i in range(size):
            b = np.random.randint(100, 256)
            g = np.random.randint(100, 256)
            r = np.random.randint(100, 256)
            self.colors.append((b,g,r))


    def _transform_correct(self):
        for s in self.seeds:
            idx = self.correct[s]-1
            self.correct[s] = self.sinks[idx]
