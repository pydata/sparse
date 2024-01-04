import xml.etree.ElementTree as et

import numpy as np


def transform(a, b, c, d, e, f):
    return f"matrix({a},{b},{c},{d},{e},{f})"


def fill(rs):
    """Generates opacity at random, weighted a bit toward 0 and 1"""
    x = rs.choice(np.arange(5), p=[0.3, 0.2, 0.0, 0.2, 0.3]) / 4
    return f"fill-opacity:{x:.1f}"


rs = np.random.RandomState(1)

colors = {
    "orange": "fill:rgb(241,141,59)",
    "blue": "fill:rgb(69,155,181)",
    "grey": "fill:rgb(103,124,131)",
}

s = 10  # face size
offset_x = 10  # x margin
offset_y = 10  # y margin
b = np.tan(np.deg2rad(30))  # constant for transformations


# reused attributes for small squares
kwargs = {"x": "0", "y": "0", "width": f"{s}", "height": f"{s}", "stroke": "white"}

# large white squares for background
bg_kwargs = {**kwargs, "width": f"{5*s}", "height": f"{5*s}", "style": "fill:white;"}


root = et.Element(
    "svg",
    **{
        "width": f"{s * 10 + 2 * offset_x}",
        "height": f"{s * 20 + 2 * offset_y}",
        "viewbox": f"0 0 {s * 10 + 2 * offset_x} {s * 20 + 2 * offset_y}",
        "version": "1.1",
        "style": "fill-rule:evenodd;clip-rule:evenodd;stroke-linejoin:round;stroke-miterlimit:2;",
        "xmlns": "http://www.w3.org/2000/svg",
        "xmlns:xlink": "http://www.w3.org/1999/xlink",
        "xml:space": "preserve",
        "xmlns:serif": "http://www.serif.com/",
        "class": "align-center",
    },
)


# face 1 (left, orange)
et.SubElement(
    root,
    "rect",
    transform=transform(1, b, 0, 1, 5 * s + offset_x, offset_y),
    **bg_kwargs,
)
for i, j in np.ndindex(5, 5):
    et.SubElement(
        root,
        "rect",
        style=f"{colors['orange']};{fill(rs)};",
        transform=transform(1, b, 0, 1, (i + 5) * s + offset_x, (i * b + j) * s + offset_y),
        **kwargs,
    )

# face 2 (top, orange)
et.SubElement(
    root,
    "rect",
    transform=transform(1, b, -1, b, 5 * s + offset_x, 5 * s + offset_y),
    **bg_kwargs,
)
for i, j in np.ndindex(5, 5):
    et.SubElement(
        root,
        "rect",
        style=f"{colors['orange']};{fill(rs)};",
        transform=transform(
            1,
            b,
            -1,
            b,
            (i - j + 5) * s + offset_x,
            (i * b + j * b + 5) * s + offset_y,
        ),
        **kwargs,
    )

# face 3 (left, blue)
for y2 in (5 + b * 5, 10 + b * 5):
    et.SubElement(
        root,
        "rect",
        transform=transform(1, b, 0, 1, offset_x, y2 * s + offset_y),
        **bg_kwargs,
    )
    for i, j in np.ndindex(5, 5):
        et.SubElement(
            root,
            "rect",
            style=f"{colors['blue']};{fill(rs)};",
            transform=transform(1, b, 0, 1, i * s + offset_x, (i * b + j + y2) * s + offset_y),
            **kwargs,
        )

# face 4 (right, grey)
et.SubElement(
    root,
    "rect",
    transform=transform(1, -b, 0, 1, 5 * s + offset_x, (10 * b + 5) * s + offset_y),
    **bg_kwargs,
)
for i, j in np.ndindex(5, 5):
    et.SubElement(
        root,
        "rect",
        style=f"{colors['grey']};{fill(rs)};",
        transform=transform(1, -b, 0, 1, (i + 5) * s + offset_x, ((10 - i) * b + j + 5) * s + offset_y),
        **kwargs,
    )

et.ElementTree(root).write("logo.svg", encoding="UTF-8")
