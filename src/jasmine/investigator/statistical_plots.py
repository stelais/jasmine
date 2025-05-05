"""Plots for histogram and cumulative distribution function of the data."""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def cumulative_distribution_presentation_plotter(working_data_, string_LX, string_LO,
                                    title=None, xmax_linear=40, what_you_plotting='Planet Mass', working_data_2_=None):
    # Scores in ascending order:
    LX_standardized_residual_order = working_data_.sort_values(f'{string_LX}')[f'{string_LX}']
    if working_data_2_ is not None:
        LO_standardized_residual_order = working_data_2_.sort_values(f'{string_LO}')[f'{string_LO}']
    else:
        LO_standardized_residual_order = working_data_.sort_values(f'{string_LO}')[f'{string_LO}']

    # Inserting 0 and 1s
    LX_standardized_residual_order = np.insert(LX_standardized_residual_order, 0, 0)
    LX_standardized_residual_order = np.insert(LX_standardized_residual_order, len(LX_standardized_residual_order), 1)
    LO_standardized_residual_order = np.insert(LO_standardized_residual_order, 0, 0)
    LO_standardized_residual_order = np.insert(LO_standardized_residual_order, len(LO_standardized_residual_order), 1)

    # Defining CDF values
    step_LX = 1 / (len(LX_standardized_residual_order) - 1)
    LX_cdf = np.arange(0, 1 + step_LX / 2, step_LX)
    step_LO = 1 / (len(LO_standardized_residual_order) - 1)
    LO_cdf = np.arange(0, 1 + step_LO / 2, step_LO)

    # Create subplots for two different types of plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Logarithmic scale plot
    ax1.step(LX_standardized_residual_order, LX_cdf, label=f'Parallax: {len(LX_standardized_residual_order) - 2}')
    ax1.step(LO_standardized_residual_order, LO_cdf,
             label=f'Parallax + Orbital Motion: {len(LO_standardized_residual_order) - 2}', color='orange')
    ax1.set_xscale('log')
    ax1.set_ylim([0, 1 + 1e-3])
    ax1.set_xlabel(fr'{what_you_plotting}')
    ax1.set_ylabel('Cumulative Distribution')
    ax1.set_title(f'LOG SCALE | {title}' if title else 'Cumulative Distribution - Log Scale')
    ax1.legend()

    # Linear scale plot with x-range limited to xmax_linear=40 default
    ax2.step(LX_standardized_residual_order, LX_cdf, label=f'Parallax: {len(LX_standardized_residual_order) - 2}')
    ax2.step(LO_standardized_residual_order, LO_cdf,
             label=f'Parallax + Orbital Motion: {len(LO_standardized_residual_order) - 2}', color='orange')
    ax2.set_xlim([-1e-5, xmax_linear])
    ax2.set_ylim([-1e-5, 1 + 1e-3])
    ax2.set_xlabel(fr'{what_you_plotting}')
    ax2.set_ylabel('Cumulative Distribution')
    ax2.set_title(f'LINEAR SCALE + ZOOM | {title}' if title else 'ZOOM - Cumulative Distribution - Linear Scale')
    ax2.legend()

    # Show the plots
    plt.tight_layout()
    plt.savefig(f"{str({what_you_plotting}).replace(' ','_').lower()}.png", dpi=300)


def histogram_presentation_plotter(working_data_, string_LX, string_LO,
                                   title=None, xmax_linear=40, what_you_plotting='Planet Mass', bins=30, working_data_2_=None):
    # Extract data for histograms
    LX_data = working_data_[string_LX].dropna()
    if working_data_2_ is not None:
        LO_data = working_data_2_[string_LO].dropna()
    else:
        LO_data = working_data_[string_LO].dropna()

    # Create subplots for two different types of plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # Logarithmic scale plot
    ax1.hist(LX_data, bins=bins, alpha=0.4, label=f'Parallax: {len(LX_data)}', log=True)
    ax1.hist(LO_data, bins=bins, alpha=0.4, label=f'Parallax + Orbital Motion: {len(LO_data)}', log=True)
    ax1.set_xscale('log')
    ax1.set_xlabel(what_you_plotting)
    ax1.set_ylabel('Count')
    ax1.set_title(f'LOG SCALE | {title}' if title else 'Histogram - Log Scale')
    ax1.legend()

    # Linear scale plot with x-range limited to xmax_linear
    ax2.hist(LX_data, bins=bins, alpha=0.4, label=f'Parallax: {len(LX_data)}')
    ax2.hist(LO_data, bins=bins, alpha=0.4, label=f'Parallax + Orbital Motion: {len(LO_data)}')
    ax2.set_xlim([-xmax_linear, xmax_linear])
    ax2.set_xlabel(what_you_plotting)
    ax2.set_ylabel('Count')
    ax2.set_title(f'LINEAR SCALE + ZOOM | {title}' if title else 'Histogram - Linear Scale')
    ax2.legend()

    # Show the plots
    plt.tight_layout()
    plt.savefig(f"{what_you_plotting.replace(' ', '_').lower()}_histogram.png", dpi=300)
    plt.show()


