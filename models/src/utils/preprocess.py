import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize
from typing import Any

def compute_residuals(
    x: np.ndarray,
    mu: float,
    phi: np.ndarray,
    theta: np.ndarray
) -> np.ndarray:
    T = len(x)
    p, q = len(phi), len(theta)
    eps = np.zeros(T)

    for t in range(T):
        idx_p = min(t, p)
        idx_q = min(t, q)
        eps[t] = (
            x[t] - mu 
            - phi[:idx_p] @ x[t - idx_p:t][::-1] 
            - theta[:idx_q] @ eps[t - idx_q:t][::-1]
        )
    
    return eps

def whiten_matrix_AR(
    X: np.ndarray,
    order: int = 1,
    by_patient: bool = True
) -> dict[str, Any]:
    n_patients, time_len, n_features = X.shape

    whitened_X = np.zeros_like(X)
    whitened_X[:, :order, :] = X[:, :order, :]
    features = X

    coeffs = np.zeros((n_patients, n_features, order))

    intercepts = np.zeros((n_patients, n_features))

    for i in range(n_features):
        if by_patient:
            for j in range(n_patients):
                x = features[j:j+1, :, i]
                lag_X = sliding_window_view(
                    x,
                    window_shape=order,
                    axis=1
                )
                rows = lag_X[:, :-1, ::-1].reshape(-1, order)
                rows = np.hstack([rows, np.ones((rows.shape[0], 1))])
                targets = x[:, order:].reshape(-1, 1)

                coeff = np.linalg.lstsq(rows, targets)[0]

                residuals = targets - rows @ coeff

                whitened_X[j, order:, i] = residuals.flatten()
                coeffs[j, i, :] = coeff[:order].flatten()
                intercepts[j, i] = coeff[order, 0]
        else:
            x = features[:, :, i]
            lag_X = sliding_window_view(
                x,
                window_shape=order,
                axis=1
            )
            rows = lag_X[:, :-1, ::-1].reshape(-1, order)
            rows = np.hstack([rows, np.ones((rows.shape[0], 1))])
            targets = x[:, order:].reshape(-1, 1)

            coeff = np.linalg.lstsq(rows, targets)[0]

            residuals = targets - rows @ coeff

            whitened_X[:, order:, i] = residuals.reshape(n_patients, time_len - order)
            coeffs[:, i, :] = np.tile(coeff[:order], n_patients).T
            intercepts[:, i] = coeff[order, 0]
    
    return {
        "X": whitened_X,
        "coeffs": coeffs,
        "intercepts": intercepts
    }

def whiten_matrix_ARMA(
    X: np.ndarray,
    order: int = 1,
    by_patient: bool = True,
    start_params: np.ndarray | None = None
) -> dict[str, Any]:
    def arma_nll(
        params: np.ndarray,
        x: np.ndarray,
        q: int
    ) -> float:
        mu = params[0]
        phi = params[1:q+1]
        theta = params[q+1:]
        T = len(x)

        eps = np.zeros(T)
        for t in range(T):
            idx = min(t, q)
            eps[t] = (
                x[t] 
                - mu 
                - phi[:idx] @ x[t - idx:t][::-1]
                - theta[:idx] @ eps[t - idx:t][::-1]
            )
        
        return np.sum(eps ** 2)
    
    def arma_nll_pooled(
        params: np.ndarray,
        patient_series: list[np.ndarray],
        q: int
    ) -> float:
        phi = params[:q]
        theta = params[q:]
        total_sse = 0

        for x in patient_series:
            mu = np.mean(x) * (1 - np.sum(phi))
            T = len(x)
            eps = np.zeros(T)

            for t in range(T):
                idx = min(t, q)
                eps[t] = (
                    x[t] 
                    - mu 
                    - phi[:idx] @ x[t - idx:t][::-1]
                    - theta[:idx] @ eps[t - idx:t][::-1]
                )
        
            total_sse += np.sum(eps ** 2)
        return total_sse

    n_patients, _, n_features = X.shape

    whitened_X = np.zeros_like(X)
    whitened_X[:, :order, :] = X[:, :order, :]
    correction_X = np.zeros_like(X)
    features = X

    ar_coeffs = np.zeros((n_patients, n_features, order))
    ma_coeffs = np.zeros((n_patients, n_features, order))

    intercepts = np.zeros((n_patients, n_features))

    for i in range(n_features):
        if by_patient:
            for j in range(n_patients):
                x = features[j, :, i]
                if start_params is None:
                    x0 = np.zeros(1 + 2 * order)   
                    x0[0] = np.mean(x)
                else:
                    x0 = start_params[i, :]
                    x0 = np.concat([np.asarray([np.mean(x)]), x0], axis=0)

                result = minimize(
                    arma_nll, 
                    x0, 
                    args=(x, order),
                    method="Nelder-Mead",
                    options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-4}
                )
                mu = result.x[0]
                phi = result.x[1:order+1]
                theta = result.x[order+1:]

                residuals = compute_residuals(
                    x,
                    mu,
                    phi,
                    theta
                )
                
                whitened_X[j, order:, i] = residuals[order:]
                correction_X[j, order:, i] = x[order:] - residuals[order:]
                ar_coeffs[j, i, :] = phi
                ma_coeffs[j, i, :] = theta
                intercepts[j, i] = mu
        else:
            patient_series = [features[j, :, i] for j in range(n_patients)]

            x0 = np.zeros(2 * order)
            result = minimize(
                arma_nll_pooled,
                x0,
                args=(patient_series, order),
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6}
            )

            shared_phi = result.x[:order]
            shared_theta = result.x[order:]

            for j in range(n_patients):
                x = features[j, :, i]
                mu = np.mean(x) * (1 - np.sum(shared_phi))

                residuals = compute_residuals(
                    x,
                    mu,
                    shared_phi,
                    shared_theta
                )

                whitened_X[j, order:, i] = residuals[order:]
                correction_X[j, order:, i] = x[order:] - residuals[order:]
                ar_coeffs[j, i, :] = shared_phi
                ma_coeffs[j, i, :] = shared_theta
                intercepts[j, i] = mu
    
    return {
        "X": whitened_X,
        "correction": correction_X,
        "ar_coeffs": ar_coeffs,
        "ma_coeffs": ma_coeffs,
        "intercepts": intercepts
    }

def whiten_by_patient_ARMA(
    features: np.ndarray,
    results: dict[str, np.ndarray]
) -> np.ndarray:
    whitened_X = np.zeros_like(features)
    n_patients, n_features = features.shape[0], features.shape[2]

    ar_coeffs = results["ar_coeffs"]
    ma_coeffs = results["ma_coeffs"]
    intercepts = results["intercepts"]
    order = ar_coeffs.shape[2]

    whitened_X[:, :order, :] = features[:, :order, :]

    for i in range(n_features):
        for j in range(n_patients):
            x = features[j, :, i]
            mu = intercepts[j, i]
            phi = ar_coeffs[j, i, :]
            theta = ma_coeffs[j, i, :]

            residuals = compute_residuals(
                x,
                mu,
                phi,
                theta
            )

            whitened_X[j, order:, i] = residuals[order:]
    
    return whitened_X

def invert_AR_whitening(
    X: np.ndarray,
    whitened_X: np.ndarray,
    ar_results: dict[str, np.ndarray],
    window: int,
    horizon: int
) -> np.ndarray:
    phi = ar_results["coeffs"]
    intercepts = ar_results["intercepts"]

    n_patients = whitened_X.shape[0]
    
    p = phi.shape[2]

    residuals = np.copy(whitened_X)
    targets = np.copy(X)

    if len(whitened_X.shape) < 3:
        targets = np.expand_dims(targets, axis=-1)
        residuals = np.expand_dims(residuals, axis=-1)
        phi = np.expand_dims(phi[:, -1, :], axis=1)
        intercepts = np.expand_dims(intercepts[:, -1], axis=1)
        n_features = 1
    else:
        n_features = residuals.shape[2]

    X_forecast = np.zeros((n_patients, horizon, n_features))

    for t in range(window, window + horizon):
        for i in range(n_patients):
            for j in range(n_features):
                val = intercepts[i, j]

                for k in range(p):
                    lag = t - 1 - k
                    if lag < window:
                        val += phi[i, j, k] * targets[i, lag, j]
                    else:
                        val += phi[i, j, k] * X_forecast[i, lag - window, j]
                
                val += residuals[i, t, j]
                X_forecast[i, t - window, j] = val

    return X_forecast

def invert_ARMA_whitening(
    X: np.ndarray,
    whitened_X: np.ndarray,
    arma_results: dict[str, np.ndarray],
    window: int,
    horizon: int
) -> np.ndarray:
    """
    Provides a forecast on an inverted whitening across a horizon.

    Params:
    X : np.ndarray : the original target values
    whitened_X : np.ndarray : the residuals to be forecast from (y - arma(x))
    arma_results : dict : dictionary of ARMA model parameters
    window : int : the index to end the window for the forecast
    horizon : int : how far from the window to forecast
    """
    phi = arma_results["ar_coeffs"]
    theta = arma_results["ma_coeffs"]
    intercepts = arma_results["intercepts"]

    n_patients = whitened_X.shape[0]
    
    p, q = phi.shape[2], theta.shape[2]

    residuals = np.copy(whitened_X)
    targets = np.copy(X)

    if len(whitened_X.shape) < 3:
        targets = np.expand_dims(targets, axis=-1)
        residuals = np.expand_dims(residuals, axis=-1)
        phi = np.expand_dims(phi[:, -1, :], axis=1)
        theta = np.expand_dims(theta[:, -1, :], axis=1)
        intercepts = np.expand_dims(intercepts[:, -1], axis=1)
        n_features = 1
    else:
        n_features = residuals.shape[2]

    # Create the forecast from the residuals
    X_forecast = np.zeros((n_patients, horizon, n_features))

    for t in range(window, window + horizon):
        for i in range(n_patients):
            for j in range(n_features):
                val = intercepts[i, j]

                for k in range(p):
                    lag = t - 1 - k
                    if lag < window:
                        val += phi[i, j, k] * targets[i, lag, j]
                    else:
                        y_hat = (
                            X_forecast[i, lag - window, j]
                            + residuals[i, lag, j]
                        )
                        val += phi[i, j, k] * y_hat
                
                for k in range(q):
                    lag = t - 1 - k
                    val += theta[i, j, k] * residuals[i, lag, j]
                
                X_forecast[i, t - window, j] = val

    # Add the forecast in to recover the true prediction
    X_true = X_forecast + residuals[:, -horizon:, :]

    return X_true