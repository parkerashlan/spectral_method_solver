from functions.gaussian import gaussian2d


def init_functions():
    w0 = lambda x, y: (
        gaussian2d(
            x,
            y,
            amplitude=5.0,
            center_x=0.25,
            center_y=0.25,
            sigma_x=(1 / 128) ** 0.5,
            sigma_y=(1 / 128) ** 0.5,
        )
        + gaussian2d(
            x,
            y,
            amplitude=5.0,
            center_x=-0.25,
            center_y=-0.25,
            sigma_x=(1 / 128) ** 0.5,
            sigma_y=(1 / 128) ** 0.5,
        )
        - gaussian2d(
            x,
            y,
            amplitude=5.0,
            center_x=-0.25,
            center_y=0.25,
            sigma_x=(1 / 128) ** 0.5,
            sigma_y=(1 / 128) ** 0.5,
        )
        - gaussian2d(
            x,
            y,
            amplitude=5.0,
            center_x=0.25,
            center_y=-0.25,
            sigma_x=(1 / 128) ** 0.5,
            sigma_y=(1 / 128) ** 0.5,
        )
    )

    j0 = lambda x, y: (
        -gaussian2d(
            x,
            y,
            amplitude=5.0,
            center_x=0.25,
            center_y=0.25,
            sigma_x=(1 / 128) ** 0.5,
            sigma_y=(1 / 128) ** 0.5,
        )
        - gaussian2d(
            x,
            y,
            amplitude=5.0,
            center_x=-0.25,
            center_y=-0.25,
            sigma_x=(1 / 128) ** 0.5,
            sigma_y=(1 / 128) ** 0.5,
        )
        - gaussian2d(
            x,
            y,
            amplitude=5.0,
            center_x=-0.25,
            center_y=0.25,
            sigma_x=(1 / 128) ** 0.5,
            sigma_y=(1 / 128) ** 0.5,
        )
        - gaussian2d(
            x,
            y,
            amplitude=5.0,
            center_x=0.25,
            center_y=-0.25,
            sigma_x=(1 / 128) ** 0.5,
            sigma_y=(1 / 128) ** 0.5,
        )
    )
    # j0 = lambda x, y: x * 0 + y * 0
    return w0, j0
