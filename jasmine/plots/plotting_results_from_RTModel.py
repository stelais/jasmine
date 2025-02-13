from bokeh.models import HoverTool, ColumnDataSource, Whisker, BoxZoomTool, ResetTool, PanTool, SaveTool, WheelZoomTool
from bokeh.plotting import figure
from bokeh.palettes import Category10

from jasmine.classes_and_files_reader.new_gullsrges_reader import whole_columns_lightcurve_reader


def plotting_mass_ratio(q_and_s_summary_df, model_type, x_range=None, y_range=None):
    if model_type == 'LS':
        title = f'Mass-ratio - 2L1S ({model_type})'
    elif model_type == 'LX':
        title = f'Mass-ratio - 2L1S+parallax ({model_type})'
    elif model_type == 'LO':
        title = f'Mass-ratio - 2L1S+parallax+OM ({model_type})'
    else:
        raise ValueError(f'model_type {model_type} should be LS, LX or LO')
    source = ColumnDataSource(data=dict(
        x=q_and_s_summary_df['true_q'],
        y=q_and_s_summary_df[f'{model_type}_q'],
        upper=q_and_s_summary_df[f'{model_type}_q'] + q_and_s_summary_df[f'{model_type}_q_err'],
        lower=q_and_s_summary_df[f'{model_type}_q'] - q_and_s_summary_df[f'{model_type}_q_err'],
        desc=q_and_s_summary_df.index,
    ))
    hover = HoverTool(tooltips=[
        ("(true,model)", "($x, $y)"),
        ("LC#", "@desc")
    ])
    if x_range is None:
        p = figure(width=350, height=350, title=title, x_axis_label='true q', y_axis_label=f'{model_type} q',
                   x_axis_type='log', y_axis_type='log',
                   tools=[hover, BoxZoomTool(), ResetTool(), PanTool(), SaveTool(), WheelZoomTool()])
    else:
        p = figure(width=350, height=350, title=title, x_axis_label='true q', y_axis_label=f'{model_type} q',
                   x_axis_type='log', y_axis_type='log',
                   tools=[hover, BoxZoomTool(), ResetTool(), PanTool(), SaveTool(), WheelZoomTool()], x_range=x_range, y_range=y_range)
    largest_true_q = q_and_s_summary_df['true_q'].max()
    smallest_true_q = q_and_s_summary_df['true_q'].min()
    p.line([smallest_true_q, largest_true_q + 1e-6], [smallest_true_q, largest_true_q + 1e-6], line_width=1, line_color='red')
    p.scatter('x', 'y', size=5, fill_alpha=0.6, source=source)
    whisker_errorbar = Whisker(source=source, base="x", upper="upper", lower="lower",
                               line_width=1.0, line_alpha=1.0)  # level="overlay",
    p.add_layout(whisker_errorbar)
    return p


def plotting_separation(q_and_s_summary_df, model_type, x_range=None, y_range=None):
    if model_type == 'LS':
        title = f'Separation - 2L1S ({model_type})'
    elif model_type == 'LX':
        title = f'Separation - 2L1S+parallax ({model_type})'
    elif model_type == 'LO':
        title = f'Separation - 2L1S+parallax+OM ({model_type})'
    else:
        raise ValueError(f'model_type {model_type} should be LS, LX or LO')
    source = ColumnDataSource(data=dict(
        x=q_and_s_summary_df['true_s'],
        y=q_and_s_summary_df[f'{model_type}_s'],
        upper=q_and_s_summary_df[f'{model_type}_s'] + q_and_s_summary_df[f'{model_type}_s_err'],
        lower=q_and_s_summary_df[f'{model_type}_s'] - q_and_s_summary_df[f'{model_type}_s_err'],
        desc=q_and_s_summary_df.index,
    ))
    hover = HoverTool(tooltips=[
        ("(true,model)", "($x, $y)"),
        ("LC#", "@desc")
    ])
    if x_range is None:
        p = figure(width=350, height=350, title=title, x_axis_label='true s', y_axis_label=f'{model_type} s',
                   tools=[hover, BoxZoomTool(), ResetTool(), PanTool(), SaveTool()])
    else:
        p = figure(width=350, height=350, title=title, x_axis_label='true s', y_axis_label=f'{model_type} s',
                   tools=[hover, BoxZoomTool(), ResetTool(), PanTool(), SaveTool()], x_range=x_range, y_range=y_range)
        # add a circle renderer with vectorized colors and sizes
    largest_true_s = q_and_s_summary_df['true_s'].max()
    smallest_true_s = q_and_s_summary_df['true_s'].min()
    p.line([smallest_true_s, largest_true_s + 1e-6], [smallest_true_s, largest_true_s + 1e-6], line_width=1, line_color='red')
    p.scatter('x', 'y', size=5, fill_alpha=0.6, source=source)
    whisker_errorbar = Whisker(source=source, base="x", upper="upper", lower="lower",
                               line_width=1.0, line_alpha=1.0)  # level="overlay",
    p.add_layout(whisker_errorbar)
    return p


def plotting_pien(parallax_summary_df, model_type, x_range=None, y_range=None):
    if model_type == 'LX':
        title = f'piEN - 2L1S+parallax ({model_type})'
    elif model_type == 'LO':
        title = f'piEN - 2L1S+parallax+OM ({model_type})'
    else:
        raise ValueError(f'model_type {model_type} should be LS, LX or LO')
    source = ColumnDataSource(data=dict(
        x=parallax_summary_df['true_piEN'],
        y=parallax_summary_df[f'{model_type}_piEN'],
        upper=parallax_summary_df[f'{model_type}_piEN'] + parallax_summary_df[f'{model_type}_piEN_error'],
        lower=parallax_summary_df[f'{model_type}_piEN'] - parallax_summary_df[f'{model_type}_piEN_error'],
        desc=parallax_summary_df.index,
    ))
    hover = HoverTool(tooltips=[
        ("(true,model)", "($x, $y)"),
        ("LC#", "@desc")
    ])
    if x_range is None:
        p = figure(width=350, height=350, title=title, x_axis_label='true pien', y_axis_label=f'{model_type} pien',
                   tools=[hover, BoxZoomTool(), ResetTool(), PanTool(), SaveTool()])
    else:
        p = figure(width=350, height=350, title=title, x_axis_label='true pien', y_axis_label=f'{model_type} pien',
                   tools=[hover, BoxZoomTool(), ResetTool(), PanTool(), SaveTool()], x_range=x_range, y_range=y_range)
    p = figure(width=350, height=350, title=title, x_axis_label='true pien', y_axis_label=f'{model_type} pien',
               tools=[hover, BoxZoomTool(), ResetTool(), PanTool(), SaveTool()])
    largest_true_piE = parallax_summary_df['true_piEN'].max()
    smallest_true_piE = parallax_summary_df['true_piEN'].min()
    p.line([smallest_true_piE, largest_true_piE], [smallest_true_piE, largest_true_piE],
           line_width=1, line_color='red')
    p.scatter('x', 'y', size=5, fill_alpha=0.6, source=source)
    whisker_errorbar = Whisker(source=source, base="x", upper="upper", lower="lower",
                               line_width=1.0, line_alpha=1.0)  # level="overlay",
    p.add_layout(whisker_errorbar)
    return p


def plotting_piee(parallax_summary_df, model_type, x_range=None, y_range=None):
    if model_type == 'LX':
        title = f'piEE - 2L1S+parallax ({model_type})'
    elif model_type == 'LO':
        title = f'piEE - 2L1S+parallax+OM ({model_type})'
    else:
        raise ValueError(f'model_type {model_type} should be LS, LX or LO')
    source = ColumnDataSource(data=dict(
        x=parallax_summary_df['true_piEE'],
        y=parallax_summary_df[f'{model_type}_piEE'],
        upper=parallax_summary_df[f'{model_type}_piEE'] + parallax_summary_df[f'{model_type}_piEE_error'],
        lower=parallax_summary_df[f'{model_type}_piEE'] - parallax_summary_df[f'{model_type}_piEE_error'],
        desc=parallax_summary_df.index,
    ))
    hover = HoverTool(tooltips=[
        ("(true,model)", "($x, $y)"),
        ("LC#", "@desc")
    ])
    if x_range is None:
        p = figure(width=350, height=350, title=title, x_axis_label='true piee', y_axis_label=f'{model_type} piee',
                   tools=[hover, BoxZoomTool(), ResetTool(), PanTool(), SaveTool()])
    else:
        p = figure(width=350, height=350, title=title, x_axis_label='true piee', y_axis_label=f'{model_type} piee',
                   tools=[hover, BoxZoomTool(), ResetTool(), PanTool(), SaveTool()], x_range=x_range, y_range=y_range)
    p = figure(width=350, height=350, title=title, x_axis_label='true piee', y_axis_label=f'{model_type} piee',
               tools=[hover, BoxZoomTool(), ResetTool(), PanTool(), SaveTool()])
    largest_true_piE = parallax_summary_df['true_piEN'].max()
    smallest_true_piE = parallax_summary_df['true_piEN'].min()
    p.line([smallest_true_piE, largest_true_piE], [smallest_true_piE, largest_true_piE],
           line_width=1, line_color='red')
    p.scatter('x', 'y', size=5, fill_alpha=0.6, source=source)
    whisker_errorbar = Whisker(source=source, base="x", upper="upper", lower="lower",
                               line_width=1.0, line_alpha=1.0)  # level="overlay",
    p.add_layout(whisker_errorbar)
    return p


def plotting_measured_and_true_relative_flux(subrun, field, event_id, sample_folder):
    lightcurve = whole_columns_lightcurve_reader(SubRun=subrun, Field=field, ID=event_id,
                                                 folder_path_=sample_folder,
                                                 include_ephem=True)

    # Assuming 'lightcurve' is your DataFrame
    # Create a custom mapping for observatory_code
    observatory_map = {0: 'W146', 1: 'Z087', 2: 'K213'}

    # Generate colors for each observatory_code using Category10 palette
    color_map = {0: Category10[3][0], 1: Category10[3][1], 2: Category10[3][2]}

    # Add observatory names and colors to the dataframe
    lightcurve['observatory_name'] = lightcurve['observatory_code'].map(observatory_map)
    lightcurve['color'] = lightcurve['observatory_code'].map(color_map)

    # Darken the color for the error bars
    lightcurve['dark_color'] = lightcurve['color'].apply(darken_color)

    # Create ColumnDataSource for measured flux
    source_measured = ColumnDataSource(data={
        'days': lightcurve['days'],
        'measured_relative_flux': lightcurve['measured_relative_flux'],
        'measured_relative_flux_error': lightcurve['measured_relative_flux_error'],
        'color': lightcurve['color'],
        'dark_color': lightcurve['dark_color'],
        'observatory_name': lightcurve['observatory_name']
    })

    # Create ColumnDataSource for true flux
    source_true = ColumnDataSource(data={
        'days': lightcurve['days'],
        'true_relative_flux': lightcurve['true_relative_flux'],
        'true_relative_flux_error': lightcurve['true_relative_flux_error'],
        'color': lightcurve['color'],
        'dark_color': lightcurve['dark_color'],
        'observatory_name': lightcurve['observatory_name']
    })

    # Create the first plot (measured_relative_flux)
    p1 = figure(
        title="Measured Flux with Error Bars",
        x_axis_label="Days",
        y_axis_label="Measured Relative Flux",
        height=400,
        width=800
    )

    # Plot measured flux with color based on observatory_code
    p1.scatter('days', 'measured_relative_flux', size=3, color='color', alpha=0.8, legend_field='observatory_name',
               source=source_measured)

    # Add error bars using Whiskers (following your example for upper and lower bounds)
    upper_measured = [x + e for x, e in
                      zip(lightcurve['measured_relative_flux'], lightcurve['measured_relative_flux_error'])]
    lower_measured = [x - e for x, e in
                      zip(lightcurve['measured_relative_flux'], lightcurve['measured_relative_flux_error'])]

    source_measured.add(upper_measured, 'upper')
    source_measured.add(lower_measured, 'lower')

    # Create the whiskers with darkened colors matching the data points
    whisker_measured = Whisker(source=source_measured, base="days", upper="upper", lower="lower",
                               line_width=0.5, line_color='dark_color',
                               line_alpha=0.3)  # Darker error bars with transparency
    whisker_measured.upper_head.line_color = 'dark_color'
    whisker_measured.lower_head.line_color = 'dark_color'
    p1.add_layout(whisker_measured)

    p1.grid.grid_line_alpha = 0.3
    p1.legend.location = "top_left"

    # Create the second plot (true_relative_flux)
    p2 = figure(
        title="True Flux with Error Bars",
        x_axis_label="Days",
        y_axis_label="True Relative Flux",
        height=400,
        width=800,
        x_range=p1.x_range  # Ensuring both plots share the same x-axis range
    )

    # Plot true flux with color based on observatory_code
    p2.scatter('days', 'true_relative_flux', size=3, color='color', alpha=0.8, legend_field='observatory_name',
               source=source_true)

    # Add error bars using Whiskers (following your example for upper and lower bounds)
    upper_true = [x + e for x, e in zip(lightcurve['true_relative_flux'], lightcurve['true_relative_flux_error'])]
    lower_true = [x - e for x, e in zip(lightcurve['true_relative_flux'], lightcurve['true_relative_flux_error'])]

    source_true.add(upper_true, 'upper')
    source_true.add(lower_true, 'lower')

    # Create the whiskers with darkened colors matching the data points
    whisker_true = Whisker(source=source_true, base="days", upper="upper", lower="lower",
                           line_width=0.5, line_color='dark_color',
                           line_alpha=0.3)  # Darker error bars with transparency
    whisker_true.upper_head.line_color = 'dark_color'
    whisker_true.lower_head.line_color = 'dark_color'
    p2.add_layout(whisker_true)

    p2.grid.grid_line_alpha = 0.3
    p2.legend.location = "top_left"
    return p1, p2


# Function to darken a color by adjusting its RGB components
def darken_color(hex_color, factor=0.8):
    # Convert hex to RGB
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    # Darken by multiplying RGB values
    rgb_darker = [max(int(x * factor), 0) for x in rgb]
    # Convert back to hex
    return '#{:02x}{:02x}{:02x}'.format(*rgb_darker)