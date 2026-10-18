"""
Minimal first-order Sugeno ANFIS in pure NumPy (no torch needed, so it also
runs on small devices at deployment).

Layers:
  1. Fuzzification : Gaussian membership functions per input
  2. Rule strength : product of the memberships (grid partition -> K**d rules)
  3. Normalization : w_bar = w / sum(w)
  4. Consequent    : f_r = p_r . [x, 1]   (linear in the inputs)
  5. Output        : y = sum_r w_bar_r * f_r

Hybrid learning each epoch:
  - consequent parameters  -> least squares (closed form, ridge-regularized)
  - membership parameters  -> gradient descent (Adam)
"""
import itertools
import json
import numpy as np


class ANFIS:
    def __init__(self, n_inputs=3, n_mf=3, init_sigma=0.25, seed=0):
        self.d, self.K = n_inputs, n_mf
        self.rules = np.array(list(itertools.product(range(n_mf), repeat=n_inputs)))
        self.R = len(self.rules)
        centers = np.linspace(0.0, 1.0, n_mf)
        self.c = np.tile(centers, (n_inputs, 1))                 # (d, K)
        self.s = np.full((n_inputs, n_mf), float(init_sigma))    # (d, K)
        self.theta = np.zeros((self.R, n_inputs + 1))            # consequents
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ forward --
    def _memberships(self, X):
        diff = X[:, :, None] - self.c[None, :, :]                # (N, d, K)
        return np.exp(-(diff ** 2) / (2 * self.s[None] ** 2)), diff

    def _firing(self, mu):
        w = np.ones((mu.shape[0], self.R))
        for i in range(self.d):
            w *= mu[:, i, self.rules[:, i]]
        return w

    def _design(self, X):
        mu, _ = self._memberships(X)
        w = self._firing(mu)
        wbar = w / (w.sum(axis=1, keepdims=True) + 1e-12)
        Xa = np.hstack([X, np.ones((len(X), 1))])                # (N, d+1)
        A = (wbar[:, :, None] * Xa[:, None, :]).reshape(len(X), -1)
        return A, mu, w, wbar, Xa

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        A = self._design(X)[0]
        return A @ self.theta.ravel()

    # ----------------------------------------------------------- training --
    def _fit_consequents(self, X, y, ridge=1e-6):
        A = self._design(X)[0]
        G = A.T @ A + ridge * np.eye(A.shape[1])
        self.theta = np.linalg.solve(G, A.T @ y).reshape(self.R, self.d + 1)

    def _premise_grad(self, X, y):
        A, mu, w, wbar, Xa = self._design(X)
        N = len(X)
        yhat = A @ self.theta.ravel()
        dL_dy = 2.0 * (yhat - y) / N
        f = Xa @ self.theta.T                                    # (N, R)
        S = w.sum(axis=1, keepdims=True) + 1e-12
        dL_dw = dL_dy[:, None] * (f - yhat[:, None]) / S         # (N, R)
        dc, ds = np.zeros_like(self.c), np.zeros_like(self.s)
        diff = X[:, :, None] - self.c[None]
        for i in range(self.d):
            for k in range(self.K):
                sel = self.rules[:, i] == k
                # d w_r / d mu_ik = w_r / mu_ik  (summed over rules using this MF)
                dL_dmu = (dL_dw[:, sel] * w[:, sel]).sum(axis=1) / (mu[:, i, k] + 1e-12)
                dc[i, k] = np.sum(dL_dmu * mu[:, i, k] * diff[:, i, k] / self.s[i, k] ** 2)
                ds[i, k] = np.sum(dL_dmu * mu[:, i, k] * diff[:, i, k] ** 2 / self.s[i, k] ** 3)
        return dc, ds

    def fit(self, X, y, X_val=None, y_val=None, epochs=300, lr=0.01, patience=40,
            verbose=True):
        """Hybrid learning. Keeps the parameters from the best validation epoch."""
        X, y = np.asarray(X, float), np.asarray(y, float)
        m = {k: np.zeros_like(self.c) for k in ("c", "s")}
        v = {k: np.zeros_like(self.c) for k in ("c", "s")}
        best, best_state, wait, history = np.inf, None, 0, []
        for ep in range(1, epochs + 1):
            self._fit_consequents(X, y)
            dc, ds = self._premise_grad(X, y)
            for name, g, p in (("c", dc, self.c), ("s", ds, self.s)):   # Adam step
                m[name] = 0.9 * m[name] + 0.1 * g
                v[name] = 0.999 * v[name] + 0.001 * g * g
                p -= lr * (m[name] / (1 - 0.9 ** ep)) / (np.sqrt(v[name] / (1 - 0.999 ** ep)) + 1e-8)
            self.s = np.clip(self.s, 0.05, 2.0)
            self._fit_consequents(X, y)
            tr = float(np.sqrt(np.mean((self.predict(X) - y) ** 2)))
            va = float(np.sqrt(np.mean((self.predict(X_val) - y_val) ** 2))) if X_val is not None else tr
            history.append((tr, va))
            if va < best - 1e-9:
                best, wait = va, 0
                best_state = (self.c.copy(), self.s.copy(), self.theta.copy())
            else:
                wait += 1
            if verbose and (ep % 25 == 0 or ep == 1):
                print(f"epoch {ep:4d}  train RMSE {tr:.4f}  val RMSE {va:.4f}")
            if wait >= patience:
                if verbose:
                    print(f"early stop at epoch {ep}")
                break
        self.c, self.s, self.theta = best_state
        return history

    # ------------------------------------------------------- save / load --
    def to_dict(self):
        return {"n_inputs": self.d, "n_mf": self.K, "c": self.c.tolist(),
                "s": self.s.tolist(), "theta": self.theta.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["n_inputs"], d["n_mf"])
        m.c, m.s, m.theta = (np.array(d["c"]), np.array(d["s"]), np.array(d["theta"]))
        return m

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
