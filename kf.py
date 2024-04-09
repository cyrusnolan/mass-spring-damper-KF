import numpy as np
import numpy.linalg as la

class KalmanFilter:
    def __init__(self, X0: np.ndarray,
                       P0: np.ndarray,
                        F: np.ndarray,
                        G: np.ndarray,
                        H: np.ndarray,
                        Q: np.ndarray,
                        R: np.ndarray) -> None:
        # Dynamics: x[k+1] = F*x[k] + G*u[k] + w[k], cov(w[k]) = Q
        #           z[k] = H*x[k] + v[k], cov(v[k]) = R
        self.F = F
        self.G = G
        self.H = H
        self.Q = Q
        self.R = R

        # number of states
        self.nx = len(X0)

        # initialize
        self.state_update = X0
        self.state_predict = None
        self.cov_update = P0
        self.cov_predict = None

    def predict(self, u) -> None:
        u = np.array([[u]])
        xu = self.state_update
        Pu = self.cov_update
        F = self.F
        G = self.G
        Q = self.Q

        # a priori: k|k-1
        self.state_predict = F@xu + G@u
        self.cov_predict = F@Pu@F.T + G@Q@G.T

    def update(self, z) -> None:
        xp = self.state_predict
        Pp = self.cov_predict
        H = self.H
        R = self.R

        # a posteriori k|k
        K = Pp@H.T@la.inv(H@Pp@H.T + R)
        self.state_update = xp + K@(z - H@xp)
        self.cov_update = (np.eye(self.nx) - K@H)@Pp@(np.eye(self.nx) - K@H).T + K@R@K.T

    @property
    def xu(self):
        return self.state_update
    
    @property
    def Pu(self):
        return self.cov_update
    
    @property
    def xp(self):
        return self.state_predict
    
    @property
    def Pp(self):
        return self.cov_predict
