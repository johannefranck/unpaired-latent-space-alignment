from matplotlib.colors import LinearSegmentedColormap

colors = [
    "#6ad6f2",  # Blue
    "#69b1d8",  # Himmelblå
    "#50a4da",  # Isblå
    "#44a1ee",  # Lyseblå
    "#5e80c4",  # Blågrå
    "#859bdb",  # Kornblå
    "#a882c4",  # Lavendel
    "#b28ec3",  # Lavendel
    "#d9bde1",  # Pink lavendel
    "#e0c7da",  # Lysest lavendel
]


def color_segment(name = "blue_pink", colorlist = colors, N=256):
    return LinearSegmentedColormap.from_list(name, colorlist, N=256)