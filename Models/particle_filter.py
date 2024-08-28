import jax
import jax.numpy as jnp
import jax.random as jr
from function import Function
from functools import partial


class ParticleFilter:
    def __init__(self, opt, R, Q):
        """
        This class contains Particle Filter implementation.
        Methods:
                - initial_sampling
                - weighting_method
                - time_update
                - measurement_update
                - resampling_step
        """
        self.opt = opt
        self.R = R
        self.Q = Q
        self.func = Function(opt)

    @staticmethod
    def initial_sampling(opt):
        """
        This staticmethod would generate the first set of necessary particles.
        Args:
              - opt: options
        Returns:
              - {'x': x_0, 'key': subkey}
        """
        num_particles = opt.num_particles
        num_states = opt.num_states
        seed_initial_pf = opt.seed_initial_pf
        key, subkey = jr.split(jr.PRNGKey(seed_initial_pf), 2)
        x_0 = jr.uniform(
            key=key,
            shape=(num_particles, num_states, 1),
            minval=opt.min_uniform,
            maxval=opt.max_uniform,
        )
        return {"x": x_0, "key": subkey}

    @staticmethod
    def weighting_method(x, h, R, z):
        """
        This staticmethod would generate weight of each particle.
        This staticmethod would be later used in another method for generating weights of all particles in a parallel manner.
        Args:
              - x: states
              - h: measurement_function
              - R: measurement covariance matrix
              - z: measurements given by sensors
        Returns:
              - w: weights of each particle
        """
        out_func = h(x, jnp.zeros((2, 1))).squeeze(-1)
        print(f"weighting_method: {out_func.shape}")
        z = jnp.expand_dims(z, axis=1)
        print(f"weighting_method z: {z.shape}")
        w = jnp.exp((-(z - out_func).T @ jnp.linalg.inv(R) @ (z - out_func)) / 2)
        return w

    def time_update(self, x, u, key):
        """
        This method would pass each particle through the function.
        Please be cautious in order to get extra credits, you need to use jax.vamp().
        Args:
              - x: states
              - u: control input
              - key: a key which is going to be used to generate random noise
        Returns:
              - x_next: particles after propagating through the process function.
              - subkey: next key.
        """
        key, subkey = jr.split(key, 2)
        omega = jr.multivariate_normal(
            key=key,
            mean=(jnp.zeros((self.opt.num_states))),
            cov=self.Q,
            shape=(self.opt.num_particles, 1),
        ).transpose(0, 2, 1)
        func_vmapped = jax.vmap(self.func.process_function, in_axes=(0, None, 0))
        print(f"x.shape: {x.shape}")
        x_next = func_vmapped(x, 0, omega)
        return x_next, subkey

    def measurement_update(self, x, z):
        """
        This method would generate the weights of all particles. It will use the weighting_method previously defined.
        Args:
            - x: states
            - z: sensor outputs
        Returns:
            - weigths: array of weights.
        """
        weighted_vmapped = jax.vmap(
            self.weighting_method, in_axes=(0, None, None, None)
        )
        weights = weighted_vmapped(x, self.func.measurement_function, self.R, z)
        weights = weights / jnp.sum(weights)
        return weights

    def resampling_step(self, x, weights, key):
        """
        This method would resample from particles.
        Args:
            - x: states
            - weights
            - key
        Returns:
            - x_new
            - subkey
        """
        c = jnp.cumsum(weights)
        print(f"resampling step c: {c.shape}")
        x = x.squeeze()
        print(f"resampling step x: {x.shape}")
        key, subkey = jr.split(key, 2)
        idx_new = jr.choice(
            key,
            jnp.arange(self.opt.num_particles),
            shape=(self.opt.num_particles,),
            replace=True,
            p=c,
        )
        x_new = jnp.expand_dims(x[idx_new, :], axis=-1)
        return x_new, subkey

    @partial(jax.jit, static_argnums=0)
    def __call__(self, z):
        def pf_scanner(res, ele):
            """
            - res: last output
            - ele: current element
            """
            x_prior, key = self.time_update(res["x"], 0, res["key"])
            weights = self.measurement_update(x_prior, ele)
            x_post, key = self.resampling_step(x_prior, weights, key)
            return {"x": x_post, "key": key}, {"x": x_post, "key": key}

        initial = self.initial_sampling(self.opt)
        final, result = jax.lax.scan(pf_scanner, initial, z)
        return result
