import jax.numpy as jnp
import jax
import jax.random as jr
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, opt, x_true):
        self.opt = opt
        self.x_true = x_true

    def plot(self, **kwargs):
        num_steps = int(self.opt.len_t / self.opt.T) + 1
        num_states = self.opt.num_states
        num_estimators = len(kwargs.keys())
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(4, hspace=0.5)
        axs = gs.subplots(sharex=True)
        # fig.suptitle('Sharing both axes')
        for i in range(num_states):
            axs[i].set_title(f"State {i+1}")
            axs[i].plot(jnp.arange(len(self.x_true)), self.x_true[:, i], label="x_true")
            for name, value in kwargs.items():
                axs[i].plot(
                    jnp.arange(len(self.x_true)), value[:, i, 0], label=f"{name}"
                )
            axs[i].legend()
        plt.show()

    # jnp.sqrt(jnp.mean((x_true - x_estimate)**2))

    def RMSE(self, **kwargs):
        num_states = self.opt.num_states
        for i in range(num_states):
            print(f"State {i+1}....")
            for name, value in kwargs.items():
                rmse_value = jnp.sqrt(
                    jnp.mean((self.x_true[:, i] - value[:, i].squeeze()) ** 2)
                )
                print(f"{name}: {rmse_value}")
