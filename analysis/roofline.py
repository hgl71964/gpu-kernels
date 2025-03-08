import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        type=int,
                        default=4096,
                        )
    parser.add_argument('-n',
                        type=int,
                        default=4096,
                        )
    parser.add_argument('-k',
                        type=int,
                        default=4096,
                        )
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        )

    parser.add_argument('--hw',
                        type=str,
                        default='h100',
                        )
    parser.add_argument('--plot',
                        action='store_true',
                        )

    return parser.parse_args()

def algo_intensity(m, n, k, dtype):
    return 2*m*n*k / (dtype * (m*n+m*k+n*k))

def hw_specs(hw):
    specs = {
            # TFLOPS, TB/s
            'a100': (),
            'h100': (1978.9, 3.35),
            }
    assert hw in specs, f'{hw} not in specs'
    return specs[hw]

def plot_roofline(ai, mem_bw, math_bw):
    intensities = np.logspace(-2, 5, 1000)  # Change the range as needed

    # Calculate performance limited by memory bandwidth
    memory_limited_performance = mem_bw * intensities

    # fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))

    # Plotting the memory bandwidth limited performance
    plt.plot(intensities, memory_limited_performance, label="Memory Bandwidth Limited", color='blue')

    # Plotting the peak computational performance
    plt.axhline(y=math_bw, color='red', linestyle='--', label="Peak Computational Performance")

    # plt.plot(intensities, [min(memory_limited_performance[i], math_bw) for i in range(len(intensities))], linestyle=':', label="roofline", color='green')
    plt.plot(intensities, [min(memory_limited_performance[i], math_bw) for i in range(len(intensities))], label="roofline", color='green')  # roofline

    plt.plot(ai, min(mem_bw*ai, math_bw),'ro', label='algorithm')

    # Setting the scale to logarithmic for both axes
    plt.xscale('log')
    plt.yscale('log')

    # Setting limits for the axes
    #plt.xlim(left=min(intensities), right=max(intensities))
    #plt.ylim(bottom=min(memory_limited_performance), top=math_bw * 2)

    # Adding labels and title
    plt.xlabel("Intensity (FLOPs/byte)")
    plt.ylabel("Performance (TFLOPs/s)")
    plt.title("Roofline Model")

    # Adding a grid and legend for clarity
    plt.grid(True, which="both", ls="--")
    plt.legend()

    # Display the plot
    plt.show()




def main():
    args = parse_args()
    m, n, k = args.m, args.n, args.k

    if args.dtype == 'fp16':
        dtype = 2
    elif args.dtype == 'fp8':
        dtype = 1
    else:
        raise

    ai = algo_intensity(m, n, k, dtype)
    math_bw, mem_bw = hw_specs(args.hw)

    if args.plot:
        plot_roofline(ai, mem_bw, math_bw)




if "__main__" == __name__:
    main()
