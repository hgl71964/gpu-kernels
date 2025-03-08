import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from roofline import hw_specs, algo_intensity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        type=str,
        default=None,
        help='Path to the text file containing matrix dimensions (m,n,k)')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        help='Data type (fp16 or fp8)')
    parser.add_argument('--hw',
                        type=str,
                        default='h100',
                        help='Hardware type (e.g., h100)')
    # parser.add_argument('--plot',
    #                     action='store_true',
    #                     help='Generate and display the roofline plot')
    return parser.parse_args()


def plot_roofline(problem_points, spec):
    math_bw, mem_bw = spec['math-bw'], spec['global-bw']

    intensities = np.logspace(-2, 5, 1000)  # Range for x-axis

    # Calculate performance limited by memory bandwidth
    global_mem_bound = mem_bw * intensities

    plt.figure(figsize=(10, 6))

    # Plotting the memory bandwidth limited performance
    plt.plot(intensities,
             global_mem_bound,
             label="global memory bound",
             color='blue')

    # plot l2 too, but the remaining still computed by global memory
    if 'l2-bw' in spec:
        l2_bound = intensities * spec['l2-bw']
        plt.plot(intensities, l2_bound, label="L2 Limited", color='slateblue')

    # Plotting the peak computational performance
    plt.axhline(y=math_bw,
                color='red',
                linestyle='--',
                label="Peak Computational Performance")

    # Plotting the roofline
    plt.plot(
        intensities,
        [min(global_mem_bound[i], math_bw) for i in range(len(intensities))],
        label="Roofline",
        color='green')

    # Plotting each problem point
    for i, (ai, m, n, k) in enumerate(problem_points):
        perf = min(mem_bw * ai, math_bw)
        plt.plot(ai, perf, 'o', label=f'Problem {i+1}: {m}x{n}x{k}')

    # Setting the scale to logarithmic for both axes
    plt.xscale('log')
    plt.yscale('log')

    # Adding labels and title
    plt.xlabel("Arithmetic Intensity (FLOPs/byte)")
    plt.ylabel("Performance (TFLOPs/s)")
    plt.title("Roofline Model with Multiple Matrix Problems")

    # Adding a grid and legend for clarity
    plt.grid(True, which="both", ls="--")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to make room for the legend
    plt.tight_layout()

    # Display the plot
    plt.show()


def read_matrix_sizes(file_path):
    assert file_path is not None
    matrix_sizes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Skip empty lines and comments
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Parse m, n, k values
                try:
                    values = line.split()
                    if len(values) == 3:
                        m, n, k = map(int, values)
                        matrix_sizes.append((m, n, k))
                    else:
                        print(
                            f"Warning: Skipping line with insufficient values: {line}"
                        )
                except ValueError:
                    print(f"Warning: Could not parse line: {line}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        exit(1)

    return matrix_sizes


def main():
    args = parse_args()

    if args.f is None:
        print("Error: Please provide a file path using the -f argument.")
        exit(1)

    # Read matrix dimensions from the file
    matrix_sizes = read_matrix_sizes(args.f)

    if not matrix_sizes:
        print("Error: No valid matrix dimensions found in the file.")
        exit(1)

    # Set the data type byte size
    if args.dtype == 'fp16':
        dtype = 2
    elif args.dtype == 'fp8':
        dtype = 1
    else:
        raise

    # Calculate arithmetic intensity for each problem
    problem_points = []
    for i, (m, n, k) in enumerate(matrix_sizes):
        ai = algo_intensity(m, n, k, dtype)
        problem_points.append((ai, m, n, k))
        print(
            f"Problem {i+1}: {m}x{n}x{k}, Arithmetic Intensity = {ai:.2f} FLOPs/byte"
        )

    # Get hardware specifications
    spec = hw_specs(args.hw)
    math_bw, mem_bw = spec['math-bw'], spec['global-bw']
    print(
        f"Hardware: {args.hw}, Compute: {math_bw} TFLOPs/s, Memory Bandwidth: {mem_bw} TB/s"
    )

    # Plot the roofline model if requested
    # if args.plot:
    #     plot_roofline(problem_points, mem_bw, math_bw)
    plot_roofline(problem_points, spec)


if __name__ == "__main__":
    main()
