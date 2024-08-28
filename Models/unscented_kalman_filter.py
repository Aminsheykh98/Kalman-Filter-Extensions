import jax
import jax.numpy as jnp
import jax.random as jr
from function import Function


class UnscentedKalmanFilter:
    def __init__(self, opt, R, Q):
        """
        This class contains Unscented Kalman Filter implementation.
        Methods:
                - time_update
                - measurement_update
        """
        self.opt = opt
        self.R = R
        self.Q = Q
        self.kappa = 3.0 - opt.num_states
        self.num_states = opt.num_states
        self.num_outputs = opt.num_outputs
        self.func = Function(opt)

    def time_update(self, x, u, P):
        """
        This method would generate and pass each sigma point through the function.
        Args:
              - x: states
              - u: control input
              - P: Covariance matrix from the previous step.
        Returns:
              - mu: The mean of sigma points.
              - P_new: Covariance matrix after necessary steps.
              - sigma_points_f: Set of sigma points after propagating through process function.
        """
        sigma_0 = x
        pos_chol = (self.num_states + self.kappa) * P
        pos_chol = jnp.linalg.cholesky(pos_chol).T
        one_n = x.repeat(repeats=self.num_states, axis=1) + pos_chol
        n_2n = x.repeat(repeats=self.num_states, axis=1) - pos_chol
        sigma_points = jnp.concatenate((x, one_n, n_2n), axis=1).T

        # weights
        w0 = (self.kappa) / (self.num_states + self.kappa)
        wi = (1.0) / (2 * (self.num_states + self.kappa))
        wi = jnp.full(2 * self.num_states, fill_value=wi)
        w = jnp.array([w0, *wi])

        # sigma_f
        func_vmapped = jax.vmap(self.func.process_function, in_axes=(0, None, None))
        sigma_points_f = func_vmapped(
            sigma_points, u, jnp.zeros((self.num_states, 1))
        ).squeeze()

        # Unscented transform
        mu = jnp.dot(w.reshape(1, -1), sigma_points_f).T
        P_new = jnp.zeros_like(P)
        for i in range(2 * self.num_states + 1):
            sigma_point_f = jnp.expand_dims(sigma_points_f[i, :], axis=1)
            P_new += w[i] * (sigma_point_f - mu) @ (sigma_point_f - mu).T
        P_new += self.Q
        return (mu, P_new, sigma_points_f)

    def measurement_update(self, x, P, sigma_points_f, z):
        """
        This method would generate and pass each sigma point through the  measurement function. And calculate next set of (x, P).
        Args:
              - x: states
              - P: Covariance matrix from the previous step.
              - sigma_points_f: sigma points obtained from time_update method.
              - z: sensor outputs
        Returns:
              - x_new: newly generated state.
              - P_new: Covariance matrix after necessary steps.
        """
        sigma_0 = x
        pos_chol = (self.num_states + self.kappa) * P
        pos_chol = jnp.linalg.cholesky(pos_chol).T
        one_n = x.repeat(repeats=self.num_states, axis=1) + pos_chol
        n_2n = x.repeat(repeats=self.num_states, axis=1) - pos_chol
        sigma_points = jnp.concatenate((x, one_n, n_2n), axis=1).T

        # weights
        w0 = (self.kappa) / (self.num_states + self.kappa)
        wi = (1.0) / (2 * (self.num_states + self.kappa))
        wi = jnp.full(2 * self.num_states, fill_value=wi)
        w = jnp.array([w0, *wi])

        func_measure = jax.vmap(self.func.measurement_function, in_axes=(0, None))
        sigma_points_h = func_measure(sigma_points, jnp.zeros((self.num_outputs, 1)))

        mu_measure = jnp.dot(w.reshape(1, -1), sigma_points_h.squeeze()).T
        P_measure = jnp.zeros((self.num_outputs, self.num_outputs))
        for i in range(2 * self.num_states + 1):
            sigma_point_h = sigma_points_h[i, :].reshape(-1, 1)
            # print(f'(sigma_point_h - mu_measure): {(sigma_point_h - mu_measure).shape}')
            P_measure += w[i] * (
                (sigma_point_h - mu_measure) @ (sigma_point_h - mu_measure).T
            )

        P_measure += self.R
        P_xz = jnp.zeros((self.num_states, self.num_outputs))
        for i in range(2 * self.num_states + 1):
            sigma_point_f = jnp.expand_dims(sigma_points_f[i], axis=1)
            P_xz += w[i] * (
                (sigma_point_f - x.reshape(-1, 1))
                @ (sigma_points_h[i, :] - mu_measure).T
            )

        K = P_xz @ jnp.linalg.inv(P_measure)
        x_new = x.reshape(-1, 1) + K @ (z.reshape(-1, 1) - mu_measure)
        P_new = P - K @ P_measure @ K.T

        return (x_new, P_new)

    def __call__(self, initial, z):
        def ukf_scanner(res, ele):
            """
            - res: last output
            - ele: current element
            """
            (x_next, P_next, sigma_points_f) = self.time_update(res["x"], 0, res["P"])
            (x_next, P_next) = self.measurement_update(
                x_next, P_next, sigma_points_f, ele
            )
            return {"x": x_next, "P": P_next}, {
                "x": x_next,
                "P": P_next,
            }  # (carryOver, accumulated)

        final, result = jax.lax.scan(ukf_scanner, initial, z)
        return result
