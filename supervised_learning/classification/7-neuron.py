ining the neuron
"""


import numpy as np


class Neuron():
    """
    neuron performing binary classification
    """
    def __init__(self, nx):
        """
        Class constructor for Neuron
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        getter function for Weights
        """
        return self.__W

    @property
    def b(self):
        """
        getter function for bias
        """
        return self.__b

    @property
    def A(self):
        """
        getter function for activation
        """
        return self.__A

def forward_prop(self, X):
    """
    Forward propagation function of the neuron
    """
    z = np.matmul(self.W, X) + self.b
    self.A = 1 / (1 + np.exp(-z))
    return self.__A

def cost(self, Y, A):
    """
    Calculates the cost of the model using logistic regression
    """
    m = Y.shape[1]
    cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
    return cost

def evaluate(self, X, Y):
    """
    Evaluates the neuron’s predictions
    """
    A = self.forward_prop(X)
    cost = self.cost(Y, A)
    prediction = np.round(A).astype(int)
    return prediction, cost

def gradient_descent(self, X, Y, A, alpha=0.05):
    """
    Calculates one pass of gradient descent on the neuron
    """
    m = Y.shape[1]
    dz = A - Y
    db = np.sum(dz) / m
    dw = np.matmul(X, dz.T) / m
    self.__W = self.__W - (alpha * dw).T
    self.__b = self.__b - alpha * db
    
 def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neuron
        iterations: number of iterations to train over
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")
        if graph:
            # Ensure we include the 0th and last iteration
            plotting_steps = np.arange(0, iterations + 1, step)
            if iterations % step != 0:
                plotting_steps = np.append(plotting_steps, iterations)

            # Initialize plotting_costs
            plotting_costs = []

            for iteration in range(iterations):
                A = self.forward_prop(X)
                self.gradient_descent(X, Y, A, alpha)
                if (iteration % step) == 0 or iteration == (iterations - 1):
                    cost = self.cost(Y, A)
                    plotting_costs.append(cost)
                    if verbose:
                        print(f"Cost after {iteration} iterations: {cost}")

            # Plot the costs
            plt.plot(plotting_steps, plotting_costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
