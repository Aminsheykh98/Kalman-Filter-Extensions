import jax
import jax.numpy as jnp
import jax.random as jr
from function import Function


class ExtendedKalmanFilter:
    def __init__(self, opt, R, Q):
        """
        This class contains Extended Kalman Filter implementation.
        Methods:
                - time_update
                - measurement_update
        """
        self.opt = opt
        self.R = R
        self.Q = Q
        self.func = Function(opt)
        self.A = jax.jacobian(self.func.process_function, argnums=0)
        self.W = jax.jacobian(self.func.process_function, argnums=2)
        self.H = jax.jacobian(self.func.measurement_function, argnums=0)
        self.V = jax.jacobian(self.func.measurement_function, argnums=1)

    def time_update(self, x, u, P):
        """
        This method is the time_update for your ekf. please code the necessary steps.
        Args:
              - 'x': States from previous step.
              - 'u': control input.
              - 'P': Covariance matrix from the previous step.
        Outputs:
              - 'x_next': State after time_update.
              - 'P_next': Covariance matrix after time_update.
        """
        x_next = self.func.process_function(
            x, u, jnp.zeros((self.opt.num_states, 1))
        ).squeeze(-1)
        A_ = self.A(x, u, jnp.zeros((self.opt.num_states, 1))).squeeze()
        W_ = self.W(x, u, jnp.zeros((self.opt.num_states, 1))).squeeze()
        P_next = A_ @ P @ A_.T + W_ @ self.Q @ W_.T
        return (x_next, P_next)

    def measurement_update(self, x, u, P, z):
        """
        This method is the measurement_update for your ekf. Please code the necessary steps.
        Args:
              - 'x': States after the time_update step.
              - 'u': control input.
              - 'P': Covariance matrix after time_update step.
              - 'z': sensor outputs.
        """
        zeros_mu = jnp.zeros((self.opt.num_outputs, 1))
        H_ = self.H(x, zeros_mu).squeeze()
        V_ = self.V(x, zeros_mu).squeeze()
        K = P @ H_.T @ jnp.linalg.inv(H_ @ P @ H_.T + V_ @ self.R @ V_.T)
        x_next = x + K @ (
            jnp.expand_dims(z, axis=1) - self.func.measurement_function(x, zeros_mu)
        )
        print(f"len x_next: {x_next.shape}")
        P_next = (jnp.eye(len(x_next)) - K @ H_) @ P
        return (x_next, P_next)

    def __call__(self, initial, z):
        """
        The call dunder method will run and save the states after each ekf step.
        You should implement this via scan method to gain the extra credit.
        !!! Fill the ekf_scanner function !!!
        """

        def ekf_scanner(res, ele):
            """
            - res: last output
            - ele: current element
            """
            (x_next, P_next) = self.time_update(res["x"], 0, res["P"])
            (x_next, P_next) = self.measurement_update(x_next, 0, P_next, ele)
            return {"x": x_next, "P": P_next}, {
                "x": x_next,
                "P": P_next,
            }  # (carryOver, accumulated)

        final, result = jax.lax.scan(ekf_scanner, initial, z)
        return result
