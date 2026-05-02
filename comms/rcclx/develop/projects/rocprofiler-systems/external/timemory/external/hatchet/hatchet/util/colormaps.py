# Copyright 2021-2023 University of Maryland and other Hatchet Project
# Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT


class ColorMaps:
    # RdYlGn (Default) color map
    RdYlGn = [
        "\033[38;5;196m",  # red
        "\033[38;5;208m",  # orange
        "\033[38;5;220m",  # yellow-ish
        "\033[38;5;46m",  # neon green
        "\033[38;5;34m",  # light green
        "\033[38;5;22m",  # dark green
    ]

    # BrBG color map
    BrBG = [
        "\033[38;5;94m",  # Brown
        "\033[38;5;179m",  # Light Brown
        "\033[38;5;222m",  # Cream
        "\033[38;5;116m",  # Light Turqouise
        "\033[38;5;37m",  # Blue-Green 73
        "\033[38;5;23m",  # Dark Blue-Green
    ]

    #  PiYG color map
    PiYG = [
        "\033[38;5;162m",  # Dark Pink
        "\033[38;5;176m",  # Light-ish Pink
        "\033[38;5;219m",  # Pale Pink
        "\033[38;5;149m",  # Pale Green
        "\033[38;5;70m",  # Light Forest Green
        "\033[38;5;22m",  # Dark Green
    ]

    # PRGn color map
    PRGn = [
        "\033[38;5;90m",  # Dark Purple
        "\033[38;5;140m",  # Purple
        "\033[38;5;183m",  # Light Purple
        "\033[38;5;151m",  # Pale Green
        "\033[38;5;70m",  # Forest Green
        "\033[38;5;22m",  # Dark Green
    ]

    # PuOr color map
    PuOr = [
        "\033[38;5;130m",  # Brown
        "\033[38;5;208m",  # Orange
        "\033[38;5;220m",  # Pale Orange
        "\033[38;5;189m",  # Pale Purple
        "\033[38;5;104m",  # Light-ish Purple
        "\033[38;5;57m",  # Dark Purple
    ]

    # RdBu color map
    RdBu = [
        "\033[38;5;124m",  # Maroon Red
        "\033[38;5;209m",  # Dark Orange
        "\033[38;5;224m",  # Pale Orange
        "\033[38;5;153m",  # Light Sky Blue
        "\033[38;5;75m",  # Light-ish Blue
        "\033[38;5;25m",  # Navy Blue
    ]

    # RdGy color map
    RdGy = [
        "\033[38;5;124m",  # Maroon Red
        "\033[38;5;209m",  # Dark Orange
        "\033[38;5;223m",  # Pale Orange
        "\033[38;5;251m",  # Pale Grey
        "\033[38;5;244m",  # Grey
        "\033[38;5;238m",  # Dark Grey
    ]

    # RdYlBu color map
    RdYlBu = [
        "\033[38;5;196m",  # Red
        "\033[38;5;208m",  # Orange
        "\033[38;5;220m",  # Yellow-ish
        "\033[38;5;153m",  # Light Cyan Blue
        "\033[38;5;68m",  # Seagull Blue
        "\033[38;5;24m",  # Dark Blue
    ]

    # Spectral color map
    Spectral = [
        "\033[38;5;196m",  # Red
        "\033[38;5;208m",  # Orange
        "\033[38;5;220m",  # Yellow-ish
        "\033[38;5;191m",  # Yellow-ish Green
        "\033[38;5;114m",  # Olive Green
        "\033[38;5;26m",  # Dark Blue
    ]

    def __init(self):
        self.colors = []

    def get_colors(self, colormap, invert_colormap):
        """Returns a list of colors based on the colormap and invert_colormap
        arguments.
        """

        if colormap == "RdYlGn":
            self.colors = self.RdYlGn
        elif colormap == "BrBG":
            self.colors = self.BrBG
        elif colormap == "PiYG":
            self.colors = self.PiYG
        elif colormap == "PRGn":
            self.colors = self.PRGn
        elif colormap == "PiYG":
            self.colors = self.PiYG
        elif colormap == "PuOr":
            self.colors = self.PuOr
        elif colormap == "RdBu":
            self.colors = self.RdBu
        elif colormap == "RdGy":
            self.colors = self.RdGy
        elif colormap == "RdYlBu":
            self.colors = self.RdYlBu
        elif colormap == "Spectral":
            self.colors = self.Spectral
        else:
            raise ValueError(
                self.colormap
                + " is an incorrect colormap. Select one BrBG, PiYg, PRGn,"
                + " PuOr, RdBu, RdGy, RdYlBu, RdYlGn, or Spectral."
            )

        # check if we need to invert the colormap
        if invert_colormap:
            self.colors.reverse()

        return self.colors
