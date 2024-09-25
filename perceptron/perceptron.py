import numpy as np

class Perceptron:
    def __init__(self, eta: float = 0.01, n_iter : int = 50, random_state: int = 1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y:np.ndarray):
        rgen = np.random.default_rng(seed=self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        self.errors_ = []


if __name__ == "__main__":
    percp = Perceptron()
    X = np.ones((10, 5))
    y = np.ones(10)
    percp.fit(X, y)