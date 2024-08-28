import argparse


def parse_opt():

    # Settings
    parser = argparse.ArgumentParser()
    root_data = "./"
    file_name = "Dataset.xlsx"

    # Function parameters
    parser.add_argument("--T", type=float, default="0.001", help="Sampling Time")
    parser.add_argument("--m", type=float, default="0.23", help="Pendulum mass")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--l", type=float, default="0.36", help="Pendulum length")
    parser.add_argument("--g", type=float, default="9.81", help="")
    parser.add_argument("--M", type=float, default="2.40", help="Cart mass")
    parser.add_argument("--path", type=str, default=root_data, help="Dataset directory")
    parser.add_argument(
        "--serial_batches",
        action="store_true",
        help="if true, takes images in order to make batches, otherwise takes them randomly",
    )
    parser.add_argument(
        "--file_name", type=str, default=file_name, help="Dataset file name"
    )
    parser.add_argument(
        "--min_uniform", type=float, default="-0.2", help="Minimum value for uniform"
    )
    parser.add_argument(
        "--max_uniform", type=float, default="0.2", help="Maximum value for uniform"
    )
    parser.add_argument(
        "--len_t", type=float, default="10.0", help="Duration of simulation"
    )
    parser.add_argument("--num_states", type=int, default=4, help="Number of states")
    parser.add_argument(
        "--num_outputs", type=int, default=2, help="Number of output variables"
    )

    parser.add_argument(
        "--num_particles",
        type=int,
        default=5000,
        help="Number of Particles used in PF algorithm",
    )
    parser.add_argument(
        "--seed_initial_pf",
        type=int,
        default=123,
        help="Initial seed to draw samples for PF",
    )

    parser.add_argument(
        "--alpha", type=float, default="0.1", help="Alpha value used for UKF"
    )

    args, unknowns = parser.parse_known_args()

    return args
