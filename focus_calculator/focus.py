import copy
import os
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from astropy.modeling import Model
from astropy.modeling import fitting, models
from astropy.stats import sigma_clipped_stats
from astropy.table import QTable, unique, Table
from astropy.table import vstack
from astropy.visualization import ZScaleInterval
from ccdproc import CCDData
from numpy import ma

from pandas import DataFrame
from pandas import concat
from pathlib import Path
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture

from scipy import optimize
from typing import List, Union
from mpl_toolkits.axes_grid1 import make_axes_locatable


from utils import (circular_aperture_statistics,
                   plot_sources_and_masked_data,
                   setup_logging)

plt.style.use('dark_background')

# log = logging.getLogger(__name__)


class FocusByImaging(object):

    def __init__(self,
                 debug: bool = False,
                 date_key: str = 'DATE',
                 date_time_key: str = 'DATE-OBS',
                 focus_key: str = 'TELFOCUS',
                 filename_key: str = 'FILENAME',
                 file_pattern: str = '*.fits',
                 n_brightest: int = 5,
                 saturation: Union[float, None] = None,
                 plot_results: bool = False,
                 debug_plots: bool = False) -> None:
        """Focus Calculator for any camera

        Args:
            debug (bool): If set to True will set the logger to debug level.
            date_key (str): FITS keyword name for obtaining the date from the FITS file. Default DATE.
            date_time_key (str): FITS keyword name for obtaining the date and time from the FITS file. Default DATE-OBS.
            focus_key (str): FITS keyword name for obtaining the focus value from the FITS file. Default TELFOCUS.
            filename_key (str): FITS keyword name for obtaining the file name from the FITS file. Default FILENAME.
            file_pattern (str): Pattern for searching files in the provided data path. Default *.fits.
            n_brightest (int): Number of the brightest sources to use for measuring source statistics. Default 5.
            saturation (float): Data value at which the detector saturates. Default 40000.
            plot_results (bool): If set to True will display information plots at the end. Default False.
            debug_plots (bool): If set to True will display several plots useful for debugging or viewing the process. Default False.
        """

        self.best_fwhm = None
        self.best_focus = None
        self.fitted_model = None
        self.best_image_overall = None
        self.date_key: str = date_key
        self.date_time_key: str = date_time_key
        self.focus_key: str = focus_key
        self.filename_key: str = filename_key
        self.file_pattern: str = file_pattern
        self.saturation_level = saturation
        self.debug: bool = debug
        self.debug_plots: bool = debug_plots
        self.plot_results: bool = plot_results
        self.mask_threshold: float = 1
        self.source_fwhm: float = 7.0
        self.det_threshold: float = 5.0
        self.show_mask: bool = False

        self.masked_data = None
        self.file_list: List
        self.sources_df: DataFrame
        self.log = setup_logging(debug=self.debug)

        self.n_brightest: int = n_brightest
        self.model: Model = models.Chebyshev1D(degree=6)
        self.fitter = fitting.LinearLSQFitter()
        self.scale = ZScaleInterval()
        self.color_map = copy.copy(cm.gray)
        self.color_map.set_bad(color='red')
        self.image_hdu_index = []

    def __call__(self,
                 data_path: Union[Path, str, None] = None,
                 file_list: Union[List, None] = None,
                 source_fwhm: float = 7.0,
                 det_threshold: float = 5.0,
                 mask_threshold: float = 1,
                 n_brightest: int = 5,
                 saturation_level: Union[float, None] = None,
                 show_mask: bool = False,
                 plot_results: bool = False,
                 debug_plots: bool = False,
                 print_all_data: bool = False) -> dict:
        """Runs the focus calculation routine

        This method contains all the logic to obtain the best focus, additionally you can parse the following parameters

        Args:
            data_path (Path, str, None): Optional data path to obtain files according to file_pattern. Default None.
            file_list (List, None): Optional file list with files to be used to obtain best focus. Default None.
            source_fwhm (float): Full width at half maximum to use for source detection and statistics.
            det_threshold (float): Number of standard deviation above median to use as detection threshold. Default 5.0.
            mask_threshold (float): Number of standard deviation below median to use as a threshold for masking values. Default 1.
            n_brightest (int): Number of the brightest sources to use for measuring source statistics. Default 5.
            saturation_level (float): Data value at which the detector saturates. Default 40000.
            show_mask (bool): If set to True will display masked values in red when debug_plots is also True: Default False.
            plot_results (bool): If set to True will display information plots at the end. Default False.
            debug_plots (bool): If set to True will display several plots useful for debugging or viewing the process. Default False.
            print_all_data (bool): If set to True will print the entire dataset at the end.

        Returns:
            A dictionary containing information regarding the current process.
        """
        if file_list:
            self.log.debug(f"Using provided file list containing {len(file_list)} files.")
            self.file_list = file_list
        elif data_path:
            self.data_path: Path = Path(data_path)
            self.log.debug(f"File list not provided, creating file list from path: {data_path}")
            self.file_list = sorted(self.data_path.glob(pattern=self.file_pattern))
            self.log.info(f"Found {len(self.file_list)} files at {self.data_path}")
        else:
            self.log.critical("You must provide at least a data_path or a file_list value")
            sys.exit(0)

        self.source_fwhm: float = source_fwhm
        self.det_threshold: float = det_threshold
        self.mask_threshold: float = mask_threshold
        if saturation_level:
            self.saturation_level: float = saturation_level
        self.show_mask: bool = show_mask
        self.n_brightest: int = n_brightest
        self.plot_results: bool = plot_results
        self.results = []

        best_image, peak, focus = self.get_best_image_by_peak()
        self.best_image_overall = Path(best_image)

        self.log.info(f"Processing best file: {self.best_image_overall.name}, selected by highest peak below saturation")

        aperture_stats = QTable()

        with fits.open(self.best_image_overall) as hdu:
            all_hdu_sources = []
            all_hdu_stats = []
            all_sources_count = 0
            for hdu_id in self.image_hdu_index:
                sources = self.detect_sources(data=hdu[hdu_id].data,
                                              primary_header=hdu[0].header,
                                              hdu_id=hdu_id,
                                              debug_plots=self.debug_plots)
                sources['id'] = range(all_sources_count + 1, all_sources_count + len(sources) + 1, 1)
                all_sources_count += len(sources)
                all_hdu_sources.append(sources)
                stats = circular_aperture_statistics(data=hdu[hdu_id].data,
                                                     primary_header=hdu[0].header,
                                                     sources=sources,
                                                     aperture_radius=self.source_fwhm,
                                                     filename=self.best_image_overall,
                                                     focus_key=self.focus_key,
                                                     plot=self.debug_plots)

                self.log.info(f"Creating mask for saturated peaks in {self.best_image_overall.name}[{hdu_id}] using saturation level {self.saturation_level}")
                saturation_mask = stats['max'] < self.saturation_level
                df = stats[saturation_mask].to_pandas()
                self.log.debug("Removing NaNs")
                df = df[df['fwhm'].notnull()]
                brightest = df.nlargest(self.n_brightest, 'max')
                # print(brightest.to_string())
                all_hdu_stats.append(QTable.from_pandas(brightest))

            stacked_sources = vstack(all_hdu_sources)
            aperture_stats = vstack(all_hdu_stats)
            # brightest = aperture_stats.nlargest(self.n_brightest, 'max')
            # print(aperture_stats.pprint_all())

            if self.debug_plots:  # pragma: no cover
                title = f"{self.n_brightest} Brightest Sources"
                positions = np.transpose((brightest['xcentroid'].tolist(), brightest['ycentroid'].tolist()))
                # plot_sources_and_masked_data(data=, positions=positions, title=title)

        if not aperture_stats:
            self.log.error("Unable to get inital photometry of selected sources.")
            sys.exit(0)

        all_photometry = []
        self.log.info("Starting photometry of all images using selected stars")
        filtered_by_hdu = unique(aperture_stats, keys='hdu_id')
        hdu_list = filtered_by_hdu['hdu_id'].tolist()

        for _file in self.file_list:
            self.log.info(f"Processing file: {_file}")
            with fits.open(_file) as hdu:
                for hdu_id in hdu_list:
                    table_mask = aperture_stats['hdu_id'] == hdu_id
                    sources_in_this_hdu = aperture_stats[table_mask]
                    positions = np.transpose((sources_in_this_hdu['xcentroid'], sources_in_this_hdu['ycentroid']))
                    photometry = circular_aperture_statistics(data=hdu[hdu_id].data,
                                                              primary_header=hdu[0].header,
                                                              sources=sources_in_this_hdu,
                                                              aperture_radius=self.source_fwhm,
                                                              filename=_file,
                                                              focus_key=self.focus_key,
                                                              plot=self.debug_plots)
                    if photometry is not None:
                        saturation_mask = photometry['max'] < self.saturation_level

                        photometry_df = photometry[saturation_mask].to_pandas()
                        photometry_df = photometry_df[photometry_df['fwhm'].notnull()]
                        if not photometry_df.empty:
                            all_photometry.append(photometry_df)
                        else:
                            self.log.warning("No data passed the saturation and not a NaN filter.")

        self.sources_df = concat(all_photometry).sort_values(by='focus').reset_index(level=0)
        self.sources_df['index'] = self.sources_df.index

        if self.debug_plots:   # pragma: no cover
            plt.show()

        star_ids = self.sources_df.id.unique().tolist()

        print(star_ids)

        all_stars_photometry = []
        all_focus = []
        all_fwhm = []
        for star_id in star_ids:
            star_phot = self.sources_df[self.sources_df['id'] == star_id]
            if len(star_phot) < 3:
                self.log.error(f"Not enough points to process source {star_id}, only {len(star_phot)} points measured.")
                break
            print(star_phot.to_string())
            try:
                interpolated_data = self.get_best_focus(df=star_phot)
                all_stars_photometry.append([star_phot, interpolated_data, self.best_focus])
                all_focus.append(self.best_focus)
                all_fwhm.append(self.best_fwhm)
            except ValueError as error:
                if self.debug:
                    self.log.exception(error)
                self.log.warning(error)
        mean_focus = np.mean(all_focus)
        median_focus = np.median(all_focus)
        focus_std = np.std(all_focus)
        mean_fwhm = np.mean(all_fwhm)

        with fits.open(self.best_image_overall) as best_image_hdu:
            self.best_image_fwhm = self.sources_df[self.sources_df['filename'] == self.best_image_overall.name]['fwhm'].mean()

            focus_data = []
            fwhm_data = []
            images = self.sources_df.filename.unique().tolist()
            for image in images:
                summary_df = self.sources_df[self.sources_df['filename'] == image]
                focus = summary_df.focus.unique().tolist()
                fwhm = summary_df.fwhm.tolist()
                if len(focus) == 1:
                    focus_data.append(focus[0])
                    fwhm_data.append(round(np.mean(fwhm), 10))

            self.results = {'date': best_image_hdu[0].header[self.date_key],
                            'time': best_image_hdu[0].header[self.date_time_key],
                            'mean_focus': round(mean_focus, 10),
                            'median_focus': round(median_focus, 10),
                            'focus_std': round(focus_std, 10),
                            'fwhm': round(mean_fwhm, 10),
                            'best_image_name': os.path.basename(self.best_image_overall),
                            'best_image_focus': best_image_hdu[0].header[self.focus_key],
                            'best_image_fwhm': round(self.best_image_fwhm, 10),
                            'focus_data': focus_data,
                            'fwhm_data': fwhm_data
                            }
            self.log.debug(f"Best Focus... Mean: {mean_focus}, median: {median_focus}, std: {focus_std}")

            if self.plot_results:   # pragma: no cover

                if len(self.image_hdu_index) == 1:
                    data = best_image_hdu[0].data
                else:
                    data = best_image_hdu[1].data

                fig, (ax1, ax2) = plt.subplots(1, 2)

                apertures = CircularAperture(positions=positions, r=1.5 * self.source_fwhm)

                z1, z2 = self.scale.get_limits(data)
                ax1.set_title(f"Best Image: {os.path.basename(best_image)}\nFocus: {best_image_hdu[0].header[self.focus_key]}")
                ax1.imshow(data, cmap=self.color_map, clim=(z1, z2), interpolation='nearest', origin='lower')
                apertures.plot(axes=ax1, color='lawngreen')

                best_image_df = self.sources_df[self.sources_df['filename'] == os.path.basename(best_image)]
                text_offset = 1.3 * self.source_fwhm
                for index, row in best_image_df.iterrows():
                    ax1.text(row['xcentroid'] - 2 * text_offset, row['ycentroid'] + text_offset, row['id'],
                             color='lawngreen', fontsize='large', fontweight='bold')
                for idx, (star_phot, interpolated_data, current_focus) in enumerate(all_stars_photometry):
                    star_id = star_phot.id.unique().tolist()[0]
                    ax2.plot(star_phot['focus'].tolist(), star_phot['fwhm'].tolist(), color=f"C{idx}",
                             label=f"Star ID: {star_id}", linestyle=':', alpha=0.7)
                    ax2.plot(interpolated_data[0], interpolated_data[1], color=f"C{idx}", alpha=0.8)
                    ax2.axvline(current_focus, color=f"C{idx}", alpha=0.8, linestyle='--')
                    ax2.set_xlabel("Focus Value")
                    ax2.set_ylabel('FWHM')
                ax2.axvline(mean_focus, color="lawngreen", label='Best Focus')
                ax2.set_title(f"Best Focus: {mean_focus}")
                ax2.legend(loc='best')
                plt.tight_layout()
                plt.show()
            if print_all_data:  # pragma: no cover
                print(self.sources_df.to_string())
            return self.results

    def create_mask(self):
        pass

    def detect_sources(self, data, primary_header, hdu_id: int, debug_plots: bool = False) -> QTable:
        """Detects sources in the sharpest image

        Using DAOStarFinder will detect the stellar sources in it.

        Args:
            image_path (Path): Image to make source detection. Should be a good image.
            debug_plots (bool): If set to True will display the image with the sources. Default False.

        Returns:
            An Astropy's QTable containing ids, centroids, focus value and image name.

        """

        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        self.log.debug(f"Mean: {mean}, Median: {median}, Standard Dev: {std}")

        mask = data <= (median - self.mask_threshold * std)
        self.masked_data = np.ma.masked_where(data <= (median - self.mask_threshold * std), data)

        self.log.debug(f"Show Mask: {self.show_mask}")
        if self.show_mask:  # pragma: no cover
            fig, ax = plt.subplots(figsize=(20, 15))
            ax.set_title(f"Bad Pixel Mask\nValues {self.mask_threshold} Std below median are masked")
            ax.imshow(self.masked_data, cmap=self.color_map, origin='lower', interpolation='nearest')

        daofind = DAOStarFinder(fwhm=self.source_fwhm,
                                threshold=median + self.det_threshold * std,
                                exclude_border=True,
                                brightest=None,
                                peakmax=self.saturation_level)
        sources = daofind(data - median, mask=mask)

        file_name = Path(primary_header[self.filename_key])
        if sources is not None:
            sources.add_column([primary_header[self.focus_key]], name='focus')
            sources.add_column(file_name.name, name='filename')
            sources.add_column(hdu_id, name='hdu_id')
            for col in sources.colnames:
                if col != 'filename':
                    sources[col].info.format = '%.8g'

            if debug_plots:  # pragma: no cover
                positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
                apertures = CircularAperture(positions, r=self.source_fwhm)

                z1, z2 = self.scale.get_limits(data)
                fig, ax = plt.subplots(figsize=(20, 15))
                ax.set_title(primary_header[self.filename_key])

                if self.show_mask:
                    masked_data = np.ma.masked_where(data <= (median - self.mask_threshold * std), data)
                    im = ax.imshow(masked_data, cmap=self.color_map, origin='lower', clim=(z1, z2),
                                   interpolation='nearest')
                else:
                    im = ax.imshow(data, cmap=self.color_map, origin='lower', clim=(z1, z2),
                                   interpolation='nearest')
                apertures.plot(color='blue', lw=1.5, alpha=0.5)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size="3%", pad=0.1)
                plt.colorbar(im, cax=cax)

        else:
            self.log.critical(f"Unable to detect sources in file: {primary_header[self.filename_key]}")
        self.log.info(f"Detected {len(sources)} sources in image {file_name.name} {f'and extension {hdu_id}' if hdu_id > 0 else ''}")
        return sources

    def get_best_focus(self, df: DataFrame, x_axis_size: int = 2000) -> List[np.ndarray]:
        """Obtains the best focus for a single source

        Args:
            df (DataFrame): Pandas DataFrame containing at least a 'focus' and a 'fwhm' column. The data should belong to a single source.
            x_axis_size (int): Size of the x-axis used to sample the fitted model. Is not an interpolation size.

        Returns:
            A list with the x-axis and the sampled data using the fitted model

        """
        focus_start = df['focus'].min()
        focus_end = df['focus'].max()

        x_axis = np.linspace(start=focus_start, stop=focus_end, num=x_axis_size)

        self.fitted_model = self.fitter(self.model, df['focus'].tolist(), df['fwhm'].tolist())
        modeled_data = self.fitted_model(x_axis)
        index_of_minimum = np.argmin(modeled_data)
        middle_point = x_axis[index_of_minimum]

        self.best_focus = optimize.brent(self.fitted_model, brack=(focus_start, middle_point, focus_end))
        self.best_fwhm = modeled_data[index_of_minimum]

        # fig, ax = plt.subplots()
        # ax.plot(x_axis, modeled_data, color='r')
        # ax.plot(df['focus'].tolist(), df['fwhm'].tolist(), color='g')
        # plt.show()

        return [x_axis, modeled_data]

    def get_best_image_by_peak(self) -> List:
        """Select the best image by its peak value

        The best peak must be the highest below the saturation level, therefore the data is first masked and then sorted
        by the peak.

        Returns:
            A list wit the best image according to the criteria described above.

        """
        data = []
        for f in self.file_list:
            with fits.open(f) as hdu:
                if not self.saturation_level:
                    if 'COADDS' in hdu[0].header.keys():
                        self.saturation_level = int(hdu[0].header['COADDS']) * 10000
                        self.log.info(f"Saturation level was calculated to be {self.saturation_level}")
                        self.log.warning("If you data with mixed coadds the saturation will be miscalculated!!")

                if len(hdu) == 1:
                    self.image_hdu_index = [0]
                else:
                    self.image_hdu_index = [i for i in range(len(hdu)) if isinstance(hdu[i], fits.ImageHDU)]
                for i in self.image_hdu_index:

                    mask = hdu[i].data >= self.saturation_level
                    masked_data = ma.masked_array(hdu[i].data, mask=mask)
                    np_max = ma.MaskedArray.max(masked_data)
                    if not ma.is_masked(np_max):
                        self.log.debug(
                            f"File: {os.path.basename(f)} {f'Extension {i}' if i > 0 else ''} Max: {np_max} Focus: {hdu[0].header[self.focus_key]}")
                        data.append([f, np_max, hdu[0].header[self.focus_key]])
                    else:
                        self.log.debug(f"Rejected masked value {np_max} from file {f}")

        df = DataFrame(data, columns=['file', 'peak', 'focus'])
        best_image = df.iloc[df['peak'].idxmax()]
        self.log.info(f"Best Image: {best_image.file} Peak: {best_image.peak} Focus: {best_image.focus}")

        return best_image.to_list()


if __name__ == '__main__':
    # focus = FocusByImaging(debug=True, date_key='DATE-OBS')
    # focus(data_path=Path('/Users/storres/data/noirlab/blanco/newfirm/focus/UT20230405/focus_3/'),
    #       source_fwhm=18,
    #       debug_plots=True,
    #       plot_results=True)

    focus = FocusByImaging(debug=True, saturation=40000)
    focus(data_path=Path('/Users/storres/data/noirlab/soar/tspec/UT20230408'), debug_plots=True, plot_results=True)


