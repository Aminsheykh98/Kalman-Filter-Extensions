import jax
import jax.numpy as jnp


class Function:
    def __init__(self, opt):
        self.T = opt.T
        self.M = opt.M
        self.m = opt.m
        self.l = opt.l
        self.g = opt.g

    def process_function(self, x, u, omega):
        """
        This method would propagate states through the process function.
        Args:
              - x: states
              - u: control input
              - omega: process noise
        """
        x_1_k = x[0] + self.T * x[1] + omega[0]
        x_2_k = (
            (
                (
                    u * jnp.cos(x[0])
                    - (self.M + self.m) * self.g * jnp.sin(x[0])
                    + self.m * self.l * (jnp.cos(x[0]) * jnp.sin(x[0])) * (x[1] ** 2)
                )
                / (
                    self.m * self.l * ((jnp.cos(x[0])) ** 2)
                    - (self.M + self.m) * self.l
                )
            )
            * self.T
            + x[1]
            + omega[1]
        )
        x_3_k = x[2] + x[3] * self.T + omega[2]
        x_4_k = (
            (
                (
                    u
                    + self.m * self.l * jnp.sin(x[0]) * (x[1] ** 2)
                    - self.m * self.g * jnp.cos(x[0]) * jnp.sin(x[0])
                )
                / (self.M + self.m - self.m * ((jnp.cos(x[0])) ** 2))
            )
            * self.T
            + x[3]
            + omega[3]
        )

        return jnp.array([[x_1_k], [x_2_k], [x_3_k], [x_4_k]])

    def measurement_function(self, x, v):
        """
        This method is the implementation of measurement function.
        Args:
              - x: states
              - v: measurement noise

        """
        y_1_k = x[0] + v[0]
        y_2_k = x[2] + v[1]
        return jnp.array([y_1_k, y_2_k])
