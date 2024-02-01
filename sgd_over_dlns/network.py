import numpy as np

q = lambda x : -1/2 * np.log((1-x**2)**2)

class LinearDiagonalNetwork:
    def __init__(self, alpha, lr, dim, batch_size=None, random_state=None, store_trajectory=False):
        self.u = alpha*np.ones(dim)*np.sqrt(2)
        self.v = np.zeros(dim)
        self.loss_history = [ ]
        self.lr = lr
        self.batch_size = batch_size
        self.store_trajectory = store_trajectory
        self.rng = random_state
        self.gain = 0
        
        self.beta_history = [ ]
    
    def forward(self, x):
        beta = np.multiply(self.u, self.v)
        return x.dot(beta)
    
    def loss(self, y_hat, y):
        if type(y) == np.float64:
            m = 1
        else:
            m = len(y)
        # 1/2n sum( (yi - sum(ujvjxij)) )
        return 1/(2*m) * np.linalg.norm(y-y_hat) 
    
    def train(self, x, y, iterations=2):
        batch_size = self.batch_size if self.batch_size is not None else x.shape[0]
        
        for i in range(iterations):
            x_batch, y_batch = x, y
            if self.batch_size is not None:
                ix = self.rng.choice(x.shape[0], replace=False)
                x_batch, y_batch = x[ix], y[ix]
                
            y_hat = self.forward(x_batch)
            loss = self.loss(y_hat, y_batch)
            self.loss_history.append(loss)
            
            gradL = 1/batch_size * x_batch.T.dot(y_hat-y_batch)
        
            gradu = np.multiply(self.v, gradL)
            gradv = np.multiply(self.u, gradL)
            
            self.gain += q(self.lr*gradL)

            self.u -= self.lr * gradu
            self.v -= self.lr * gradv
                        
            if self.store_trajectory:
                self.beta_history.append(np.multiply(self.u, self.v))
            
    def get_loss(self, x, y):
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return loss