try:
    import plotnine as pn
except ImportError as e:
    raise ImportError("The 'notebooks' extra is required to use this module.") from e

_shapes = ["o", "s", "D", "^", "v", "<", ">", "*", "p", "h", "8", "+", "x"]
_colors_muted = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]

_background = "#FFFFFF"
_text = "#222222"
_axis = "#666666"
_grid = "#CCCCCC"


def theme(base_size=11, base_family="DejaVu Sans"):
    return pn.theme_minimal(base_size=base_size, base_family=base_family) + pn.theme(
        panel_background=pn.element_rect(fill=_background, color=_background),
        plot_background=pn.element_rect(fill=_background, color=_background),
        panel_grid_major=pn.element_line(color=_grid, size=0.4),
        panel_grid_minor=pn.element_blank(),
        axis_text=pn.element_text(color=_text),
        axis_title=pn.element_text(weight="normal"),
        plot_title=pn.element_text(weight="normal", size=base_size),
        plot_subtitle=pn.element_text(size=base_size * 0.8),
        plot_caption=pn.element_text(size=base_size * 0.7, color=_axis),
        legend_position="right",
        legend_box_background=pn.element_blank(),
        legend_background=pn.element_blank(),
        legend_key=pn.element_blank(),
        legend_title=pn.element_text(weight="normal"),
        strip_background=pn.element_rect(fill=_background, color=_background),
        strip_text=pn.element_text(weight="normal"),
        figure_size=(6, 4),
    )


def fill(labels=None):
    if labels:
        return pn.scale_fill_manual(values=_colors_muted, labels=labels)

    return pn.scale_fill_manual(values=_colors_muted)


def color(labels=None):
    if labels:
        return pn.scale_color_manual(values=_colors_muted, labels=labels)

    return pn.scale_color_manual(values=_colors_muted)


def shape(labels=None):
    if labels:
        return pn.scale_shape_manual(values=_shapes, labels=labels)

    return pn.scale_shape_manual(values=_shapes)
