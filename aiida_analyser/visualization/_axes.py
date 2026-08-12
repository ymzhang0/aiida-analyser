"""Internal helpers for plotting into new or supplied Matplotlib axes."""


def axis_limits(limits, num_axes, parameter='ylim'):
    """Normalise one axis limit or one limit pair per subplot."""
    if limits is None:
        return [None] * num_axes

    try:
        items = list(limits)
    except TypeError as exc:
        raise TypeError(f'{parameter} must be a two-value range or one range per subplot.') from exc

    def is_nested_range(value):
        if isinstance(value, (str, bytes)):
            return False
        try:
            iter(value)
        except TypeError:
            return False
        return True

    if len(items) == 2 and not any(is_nested_range(value) for value in items):
        return [tuple(items)] * num_axes
    if len(items) != num_axes:
        raise ValueError(
            f'{parameter} must be one two-value range or contain {num_axes} ranges, one for each subplot.'
        )

    normalised = []
    for value in items:
        try:
            limit = tuple(value)
        except TypeError as exc:
            raise TypeError(f'Each {parameter} entry must be a two-value range.') from exc
        if len(limit) != 2:
            raise ValueError(f'Each {parameter} entry must contain exactly two values.')
        normalised.append(limit)
    return normalised


def plot_axes(axes, num_axes, *, plt, figsize):
    """Create plot axes or validate a user-supplied set of axes."""
    if axes is None:
        figure, axes_array = plt.subplots(1, num_axes, figsize=figsize, squeeze=False)
        return figure, axes_array.ravel()

    def flatten(value):
        if hasattr(value, 'plot') and hasattr(value, 'figure'):
            return [value]
        try:
            values = iter(value)
        except TypeError as exc:
            raise TypeError('axes must be a Matplotlib Axes or an iterable of Axes.') from exc
        return [axis for item in values for axis in flatten(item)]

    axes_array = flatten(axes)
    if len(axes_array) != num_axes:
        raise ValueError(f'axes must contain exactly {num_axes} axes, one for each selected material.')
    figure = axes_array[0].figure
    if any(axis.figure is not figure for axis in axes_array):
        raise ValueError('All supplied axes must belong to the same Matplotlib figure.')
    return figure, axes_array
