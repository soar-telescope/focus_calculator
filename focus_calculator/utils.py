import copy
import os
import logging
import sys
import numpy as np
import numpy.ma as ma
from pathlib import  Path
from argparse import ArgumentParser, Namespace

from astropy.io import fits
from astropy.table import QTable, unique
from astropy.visualization import ZScaleInterval
from ccdproc import CCDData
from logging import Logger

from matplotlib import cm, pyplot as plt
from pandas import DataFrame
from typing import Union, List

from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats

log = logging.getLogger(__name__)


def circular_aperture_statistics(data,
                                 primary_header,
                                 sources,
                                 aperture_radius: float = 10,
                                 filename: Union[Path, None] = None,
                                 filename_key: Union[str, None] = None,
                                 focus_key: str = 'TELFOCUS',
                                 plot: bool = False) -> DataFrame:
    """Obtain aperture statistics from a FITS file with point sources

    Uses CircularAperture from photutils to obtain several data but the most important is the FWHM

    Args:
        image_path (Path): An image with point sources.
        sources (QTable): Coordinates of the sources to be measured.
        aperture_radius (float): Aperture size for obtaining measurements. Default 10.
        filename_key (str): FITS keyword name for obtaining the file name from the FITS file. Default FILENAME.
        focus_key (str): FITS keyword name for obtaining the focus value from the FITS file. Default TELFOCUS.
        plot (bool): If set to True will display a plot of the image and the measured sources.

    Returns:
        A DataFrame containing the following columns: id, mean, fwhm, max, xcentroid, ycentroid and filename.

    """
    if not (filename or filename_key):
        log.error("Must provide either 'filename' or 'filename_key'")
        sys.exit(0)
    if filename_key:
        filename = Path(primary_header[filename_key])
    sources_positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(sources_positions, r=aperture_radius)

    aperture_stats = ApertureStats(data, apertures, sigma_clip=None)

    columns = ('id', 'mean', 'fwhm', 'max', 'xcentroid', 'ycentroid')
    as_table = aperture_stats.to_table(columns=columns)
    as_table['id'] = sources['id']
    as_table['focus'] = primary_header[focus_key]
    as_table['filename'] = filename.name
    as_table['hdu_id'] = sources['hdu_id']
    for col in as_table.colnames:
        if col not in ['filename']:
            as_table[col].info.format = '%.8g'

    if plot:  # pragma: no cover
        title = f"Photometry of {filename.name}"
        plot_sources_and_masked_data(data=data, positions=sources_positions, title=title)

    return as_table


def get_args(arguments: Union[List, None] = None) -> Namespace:
    """Helper function to get the console arguments using argparse.

    Args:
        arguments (List, None): Optional list of arguments. Default None.

    Returns:
        An instance of arparse's Namespace.

    """
    parser = ArgumentParser(
        description="Get best focus value using a sequence of images with "
                    "different focus value"
    )

    parser.add_argument('--data-path',
                        action='store',
                        dest='data_path',
                        default=os.getcwd(),
                        help='Folder where data is located')

    parser.add_argument('--file-pattern',
                        action='store',
                        dest='file_pattern',
                        default='*.fits',
                        help='Pattern for filtering files.')

    parser.add_argument('--focus-key',
                        action='store',
                        dest='focus_key',
                        default='TELFOCUS',
                        help='FITS header keyword to find the focus value.')

    parser.add_argument('--filename-key',
                        action='store',
                        dest='filename_key',
                        default='FILENAME',
                        help='FITS header keyword to find the current file name.')

    parser.add_argument('--brightest',
                        action='store',
                        dest='brightest',
                        type=int,
                        default=5,
                        help='Pick N-brightest sources to perform statistics. Default 5.')

    parser.add_argument('--saturation',
                        action='store',
                        dest='saturation',
                        type=int,
                        default=40000,
                        help='Saturation value for data')

    parser.add_argument('--source-fwhm',
                        action='store',
                        dest='source_fwhm',
                        default=7.0,
                        help='FWHM for source detection.')

    parser.add_argument('--detection-threshold',
                        action='store',
                        dest='detection_threshold',
                        default=6,
                        help='Number of standard deviation above median for source detection.')

    parser.add_argument('--mask-threshold',
                        action='store',
                        dest='mask_threshold',
                        default=1,
                        help='Number of standard deviation below median to mask values.')

    parser.add_argument('--plot-results',
                        action='store_true',
                        dest='plot_results',
                        help='Show a plot when it finishes the focus '
                             'calculation')

    parser.add_argument('--show-mask',
                        action='store_true',
                        dest='show_mask',
                        help='Show the image and the masked areas highlighted in red.')

    parser.add_argument('--debug',
                        action='store_true',
                        dest='debug',
                        help='Activate debug mode')

    parser.add_argument('--debug-plots',
                        action='store_true',
                        dest='debug_plots',
                        help='Show debugging plots.')

    args = parser.parse_args(args=arguments)
    return args


def plot_sources_and_masked_data(data, positions: np.ndarray, title: str = '', aperture_radius: int = 10):  # pragma: no cover
    """Helper function to plot data and sources

    Args:
        ccd (CCDData): The image to be shown.
        positions (numpy.ndarray): Array of source positions to be plotted over the image.
        title (str): Title to put to the plot. Default '' (empty string).
        aperture_radius (int): Radius size in pixels for the drawing of the sources. Default 10.

    """
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title(title)

    color_map = copy.copy(cm.gray)
    color_map.set_bad(color='green')

    scale = ZScaleInterval()
    z1, z2 = scale.get_limits(data)

    apertures = CircularAperture(positions, r=aperture_radius)
    annulus_apertures = CircularAnnulus(positions, r_in=aperture_radius + 5, r_out=aperture_radius + 10)

    ax.imshow(data, cmap=color_map, clim=(z1, z2))
    apertures.plot(color='red')
    annulus_apertures.plot(color='yellow')

    plt.tight_layout()
    plt.show()


def setup_logging(debug: bool = False, enable_astropy_logger: bool = False) -> Logger:
    """Helper function to setup the logger.

    Args:
        debug (bool): If set to True will create a logger in debug level. Default False.
        enable_astropy_logger (bool): If set to True will allow the astropy logger to remain active. Default False.

    Returns:
        A Logger instance from python's logging.

    """
    if debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger = logging.getLogger(__name__)
    logger.setLevel(level=log_level)

    console = logging.StreamHandler()
    console.setLevel(log_level)

    log_format = '[%(asctime)s][%(levelname)s]: %(message)s'
    formatter = logging.Formatter(log_format)

    console.setFormatter(formatter)

    logger.addHandler(console)

    astropy_logger = logging.getLogger('astropy')
    if enable_astropy_logger or debug:
        for handler in astropy_logger.handlers:
            astropy_logger.removeHandler(handler)
        astropy_logger.addHandler(console)
    else:
        astropy_logger.disabled = True

    return logger
