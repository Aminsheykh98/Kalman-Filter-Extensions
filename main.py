import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd

from create_load import CreateOrLoad
from argparser import parse_opt
from plot import Plotter
import Models as models
from function import Function

if __name__ == "__main__":
    opt = parse_opt()
    R = jnp.diag(jnp.array([1.2, 0.8]))
    Q = jnp.diag(jnp.array([0.9, 0.8, 0.9, 0.8]))
    create_data = CreateOrLoad(opt, R, Q)
    func = Function(opt)
    ekf = models.ExtendedKalmanFilter(opt, R, Q)
    key = jr.PRNGKey(opt.seed + 1)
    key1, key2, subkey = jr.split(key, 3)
    x_0 = jr.uniform(
        key=key1,
        shape=(opt.num_states, 1),
        minval=opt.min_uniform,
        maxval=opt.max_uniform,
    )
    v_0 = jr.multivariate_normal(key=key2, mean=jnp.zeros((opt.num_outputs)), cov=R)
    y_0 = func.measurement_function(x_0, v_0)
    P_0 = jnp.eye(4)  # * 1000

    initial_value = {"x": x_0, "P": P_0}
    result_ekf = ekf(initial_value, create_data.y)

    ukf = models.UnscentedKalmanFilter(opt, R, Q)
    result_ukf = ukf(initial_value, create_data.y)

    pf = models.ParticleFilter(opt, R, Q)
    initial_value = {"x": x_0, "key": subkey}
    result_pf = pf(create_data.y)
    x_pf = jnp.mean(result_pf["x"], axis=1)

    # To illustrate the results just pass filter's name as keyword argument and the state results like below
    plt_class = Plotter(opt, create_data.x)
    plt_class.plot(ukf=result_ukf["x"], ekf=result_ekf["x"])

    plt_class.RMSE(ukf=result_ukf["x"], ekf=result_ekf["x"])
