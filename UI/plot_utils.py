# UI/plot_utils.py

import cv2
import numpy as np


def render_image_overlay(ax, plot_area_img, plot_extent, x_scale='linear', y_scale='linear',
                         max_mesh_cells=200):
    """
    Renders an image overlay on the axes, correctly handling log-scaled axes.

    imshow always maps pixels linearly across the extent, which distorts the
    image on log-scaled axes.  For any axis that is log-scaled we use
    pcolormesh with logarithmically-spaced mesh coordinates instead, so every
    image pixel lands at the correct data-space position.

    Parameters
    ----------
    ax :             matplotlib Axes to draw on.
    plot_area_img :  RGB numpy array of the cropped plot area.
    plot_extent :    [x_min, x_max, y_min, y_max] in data coordinates.
    x_scale :        'linear' or 'log'.
    y_scale :        'linear' or 'log'.
    max_mesh_cells : maximum number of cells along the longer image dimension.
                     The image is downsampled to this size before building the
                     mesh, keeping pcolormesh fast.  200 gives a good
                     quality / performance trade-off.
    """
    h, w = plot_area_img.shape[:2]
    x_min, x_max, y_min, y_max = plot_extent

    if x_scale != 'log' and y_scale != 'log':
        # Linear on both axes – imshow works fine
        ax.imshow(plot_area_img,
                  extent=plot_extent,
                  aspect='auto',
                  alpha=0.3,
                  zorder=0,
                  interpolation='bilinear',
                  origin='upper')
        return

    # --- Downsample for performance ---
    scale = max_mesh_cells / max(h, w)
    if scale < 1.0:
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)
        plot_area_img = cv2.resize(plot_area_img, (new_w, new_h),
                                   interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    # Build mesh edge coordinates (size w+1 and h+1)
    if x_scale == 'log':
        x_edges = np.logspace(np.log10(x_min), np.log10(x_max), w + 1)
    else:
        x_edges = np.linspace(x_min, x_max, w + 1)

    if y_scale == 'log':
        y_edges = np.logspace(np.log10(y_min), np.log10(y_max), h + 1)
    else:
        y_edges = np.linspace(y_min, y_max, h + 1)

    # Image row 0 is top => flip so row 0 maps to y_min (bottom of mesh)
    img_flipped = plot_area_img[::-1, :, :]

    # Build RGBA float array with alpha=0.3
    img_rgba = np.zeros((h, w, 4), dtype=np.float32)
    img_rgba[..., :3] = img_flipped.astype(np.float32) / 255.0
    img_rgba[..., 3] = 0.3

    X, Y = np.meshgrid(x_edges, y_edges)
    dummy = np.zeros((h, w))
    mesh = ax.pcolormesh(X, Y, dummy, shading='flat', zorder=0)
    mesh.set_array(None)
    mesh.set_facecolors(img_rgba.reshape(-1, 4))
