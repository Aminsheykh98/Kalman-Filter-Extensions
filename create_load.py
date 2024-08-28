import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import os
from tqdm import tqdm
from functools import partial
from function import Function


class CreateOrLoad:
    def __init__(self, opt, R, Q):
        """
        This class would generate/load the dataset.
        """
        self.opt = opt
        self.R = R
        self.Q = Q
        is_exist = self.exists(opt.path, opt.file_name)

        if is_exist:
            self.x, self.y = self.load_data()
            print(f"x.shape: {self.x.shape}")
            print(f"y.shape: {self.y.shape}")

        else:
            self.func = Function(self.opt)
            x, y = self.make_data()
            self.x, self.y = x.squeeze(), y.squeeze()
            self.save_data(self.y, self.x, opt.path, opt.file_name)

    @staticmethod
    def exists(path, name):
        """
        This staticmethod would check whether the dataset exists or not.
        """
        is_exist = os.path.isfile(os.path.join(path, name))
        if is_exist:
            print(f"File {name} has been found in the directory")
            return True
        else:
            print(f"File {name} could not been found")
            return False

    @staticmethod
    def save_data(y, x, path, name):
        print(f"Saving dataset...")
        columns = ["y_1", "y_2", "x_1", "x_2", "x_3", "x_4"]
        y_and_x = jnp.concatenate((y.squeeze(), x.squeeze()), axis=1)
        df = pd.DataFrame(data=y_and_x, columns=columns)
        df.to_excel(os.path.join(path, name))

    @partial(jax.jit, static_argnums=0)
    def make_data(self):
        """
        This method would build data if it is not currently available in the given directory.
        """
        print(f"Building dataset...")
        key = jr.PRNGKey(self.opt.seed)
        key1, key2, subkey = jr.split(key, 3)
        x_0 = jr.uniform(
            key=key1,
            shape=(self.opt.num_states, 1),
            minval=self.opt.min_uniform,
            maxval=self.opt.max_uniform,
        )
        v_0 = jr.multivariate_normal(
            key=key2, mean=jnp.zeros((self.opt.num_outputs)), cov=self.R
        )
        y_0 = self.func.measurement_function(x_0, v_0)
        num_steps = int(self.opt.len_t / self.opt.T)
        total_keys = jr.split(subkey, num_steps)

        def system_propagate(res, key):
            """
            - res: result from the previous loop.
            - key: current key element.
            """
            key_x, key_y = jr.split(key, 2)
            omega = jr.multivariate_normal(
                key_x, mean=jnp.zeros((self.opt.num_states)), cov=self.Q
            )
            x_next = self.func.process_function(res["x"].squeeze(), 0, omega)
            v = jr.multivariate_normal(
                key_y, mean=jnp.zeros(self.opt.num_outputs), cov=self.R
            )
            y_next = self.func.measurement_function(x_next, v)
            return {"x": x_next, "y": y_next}, {
                "x": x_next,
                "y": y_next,
            }  # (carry_over, accumulation)

        final, result = jax.lax.scan(system_propagate, {"x": x_0, "y": y_0}, total_keys)
        x = jnp.concatenate((jnp.expand_dims(x_0, axis=0), result["x"]), axis=0)
        y = jnp.concatenate((jnp.expand_dims(y_0, axis=0), result["y"]), axis=0)
        return x, y

    def load_data(self):
        """
        This method would load the dataset from the given directory.
        """
        print(f"Loading dataset...")
        path_dataset = os.path.join(self.opt.path, self.opt.file_name)
        df = pd.read_excel(path_dataset)
        x = jnp.array(df.to_numpy()[:, 3:])
        y = jnp.array(df.to_numpy()[:, 1:3])
        return x, y

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return {"x": x, "y": y}

    def __len__(self):
        return len(self.x)
