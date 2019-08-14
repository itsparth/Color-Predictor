import numpy as np
from tkinter import ttk
import tkinter, random

input = np.zeros(shape=(3, 3, 1))
output = np.zeros(shape=(3, 2))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork:
    def __init__(self, sizes, loadmode=False):
        self.num_layers = len(sizes)
        self.sizes = sizes
        if loadmode:
            self.weights = np.load("weights.npy")
            self.biases = np.load("biases.npy")
        else:
            self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
            self.weights = [np.random.rand(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta):
        n = len(training_data)

        for j in range(epochs):
            print(j)
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

        np.save("weights", self.weights)
        np.save("biases", self.biases)

    def update_mini_batch(self, mini_batch, eta):

        nebla_b = [np.zeros(b.shape) for b in self.biases]
        nebla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nebla_b, delta_nebla_w = self.backprop(x, y)
            nebla_b = [nb + dnb for nb, dnb in zip(nebla_b, delta_nebla_b)]
            nebla_w = [nw + dnw for nw, dnw in zip(nebla_w, delta_nebla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nebla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nebla_b)]

    def backprop(self, x, y):
        nebla_b = [np.zeros(b.shape) for b in self.biases]
        nebla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [activation]

        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nebla_b[-1] = delta
        nebla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp

            nebla_b[-l] = delta
            nebla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nebla_b, nebla_w)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


class generateDataUI:
    def __init__(self, count):
        self.index = 0
        self.count = count - 1
        self.root = tkinter.Tk()
        self.root.title("Colour Predictor - Data Collection")
        self.root.geometry("300x100")

        color = self.getColour()
        self.label1 = ttk.Label(self.root, text="TEXT", background=color, anchor="center", font=("Calibri", 20))
        self.label1.grid(row=1, column=0, sticky="NESW", padx=5)
        self.label2 = ttk.Label(self.root, text="TEXT", background=color, foreground="white", anchor="center",
                                font=("Calibri", 20))
        self.label2.grid(row=1, column=1, sticky="NESW", padx=5)

        tkinter.Grid.columnconfigure(self.root, 0, weight=1)
        tkinter.Grid.columnconfigure(self.root, 1, weight=1)

        tkinter.Grid.rowconfigure(self.root, 1, weight=1)

        self.btn1 = ttk.Button(self.root, text="BLACK", command=self.changeColourBlack).grid(row=2, column=0, pady=5)
        self.btn2 = ttk.Button(self.root, text="WHITE", command=self.changeColourWhite).grid(row=2, column=1, pady=5)

        self.root.mainloop()

    def changeColourBlack(self):
        if self.index >= self.count:
            self.root.destroy()
            return
        global input
        global output
        input[self.index] = [[self.r / 255], [self.g / 255], [self.b / 255]]
        output[self.index] = [[1], [0]]
        self.index += 1
        colour = self.getColour()
        self.label1.config(background=colour)
        self.label2.config(background=colour)

    def changeColourWhite(self):
        if self.index >= self.count:
            self.root.destroy()
            return
        global input
        global output
        input[self.index] = [[self.r / 255], [self.g / 255], [self.b / 255]]
        output[self.index] = [[0], [1]]
        self.index += 1
        colour = self.getColour()
        self.label1.config(background=colour)
        self.label2.config(background=colour)

    def getColour(self):
        self.r = random.randint(0, 255)
        self.g = random.randint(0, 255)
        self.b = random.randint(0, 255)
        return '#%02x%02x%02x' % (self.r, self.g, self.b)


class TestUI:
    def __init__(self, array=None, epochs=20000, batch=10, eta=0.01):
        self.root = tkinter.Tk()
        self.root.title("Colour Predictor - Test")
        self.root.geometry("300x100")

        self.array = array
        self.epochs = epochs
        self.batch = batch
        self.eta = eta

        self.label0l = ttk.Label(self.root, text="", background="white", anchor="center", font=("Calibri", 2))
        self.label0r = ttk.Label(self.root, text="", background="white", anchor="center", font=("Calibri", 2))

        self.label0l.grid(row=0, column=0, sticky="NESW", padx=5, pady=5)
        self.label0r.grid(row=0, column=1, sticky="NESW", padx=5, pady=5)

        self.train_network()

        colour = self.getColour()

        self.label1 = ttk.Label(self.root, text="TEXT", background=colour, anchor="center", font=("Calibri", 20))
        self.label1.grid(row=1, column=0, sticky="NESW", padx=5)
        self.label2 = ttk.Label(self.root, text="TEXT", background=colour, foreground="white", anchor="center",
                                font=("Calibri", 20))
        self.label2.grid(row=1, column=1, sticky="NESW", padx=5)

        tkinter.Grid.columnconfigure(self.root, 0, weight=1)
        tkinter.Grid.columnconfigure(self.root, 1, weight=1)

        tkinter.Grid.rowconfigure(self.root, 1, weight=1)

        self.btn1 = ttk.Button(self.root, text="NEXT", command=self.selectBlack).grid(row=2, column=0, pady=5, columnspan=2)
        # self.btn2 = ttk.Button(self.root, text="WHITE", command=self.selectWhite).grid(row=2, column=1, pady=5)

        self.root.mainloop()

    def selectBlack(self):
        colour = self.getColour()
        self.label1.config(background=colour)
        self.label2.config(background=colour)

    def selectWhite(self):
        colour = self.getColour()
        self.label1.config(background=colour)
        self.label2.config(background=colour)

    def getColour(self):
        self.r = random.randint(0, 255)
        self.g = random.randint(0, 255)
        self.b = random.randint(0, 255)
        self.label0l.config(background="white")
        self.label0r.config(background="white")

        self.output = self.network.feedforward(np.array([[self.r / 256], [self.g / 256], [self.b / 256]]))

        if self.output[0][0] >= 0.5:
            self.label0l.config(background="black")
            print("Percentage Sure: " + str(float(round(self.output[0][0] * 10000)) / 100))
        else:
            self.label0r.config(background="black")
            print("Percentage Sure: " + str(float(round(self.output[1][0] * 10000)) / 100))

        return '#%02x%02x%02x' % (self.r, self.g, self.b)

    def train_network(self):
        array = self.array
        if array == None:
            self.network = NeuralNetwork([3, 4, 2], loadmode=True)
        else:
            self.network = NeuralNetwork([3, 4, 2])
            self.network.SGD(array, self.epochs, self.batch, self.eta)


def save_array(array):
    np.save("data", array)


def load_array():
    return np.load("data.npy")


def collectData(amount):
    global input, output
    input = np.zeros(shape=(amount, 3, 1))
    output = np.zeros(shape=(amount, 2, 1))
    generateDataUI(amount)
    array = [(i, o) for i, o in zip(input, output)]
    # save_array(array)
    return array


def main():
    TestUI()


if __name__ == '__main__':
    main()
