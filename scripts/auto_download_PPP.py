# Script for auto-downloading the necessary files to run PPP solutions in Ginan
# import boto3
import os
import re
import csv
import json
import click
import random
import ftplib
import logging
import requests
import numpy as np
from time import sleep
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse
from contextlib import contextmanager
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from gnssanalysis.gn_datetime import GPSDate
from gnssanalysis.gn_download import (
    download_product_from_cddis,
    decompress_file,
    generate_content_type,
    generate_sampling_rate,
    generate_product_filename,
    download_file_from_cddis,
    get_earthdata_credentials,
    get_earthdata_token,
    check_whether_to_download,
    attempt_url_download,
    long_filename_cddis_cutoff,
)
from gnssanalysis.gn_utils import configure_logging, ensure_folders

API_URL = "https://data.gnss.ga.gov.au/api"
CDDIS_RINEX_BASE = "https://cddis.nasa.gov/archive/gnss/data/daily"
_RINEX3_OBS_RE = re.compile(r"^[A-Z0-9]{9}_[RSU]_\d{11}_01D_\d+[SMHD]_MO\.crx\.gz$", re.IGNORECASE)

_cddis_thread_local = threading.local()


def _get_cddis_session(username: str, password: str, pool_size: int = 4) -> requests.Session:
    """Return a thread-local requests session for CDDIS, creating it on first use per thread.

    Prefers Bearer token auth (no OAuth redirects) over username/password.
    Token is read from the EARTHDATA_TOKEN environment variable (earthaccess convention).
    """
    if not hasattr(_cddis_thread_local, "session"):
        from requests.adapters import HTTPAdapter
        session = requests.Session()
        token = get_earthdata_token()
        if token:
            logging.debug("CDDIS session: using Bearer token (EARTHDATA_TOKEN)")
            session.headers["Authorization"] = f"Bearer {token}"
        else:
            logging.debug("CDDIS session: using username/password auth")
            session.auth = (username, password)
            session.max_redirects = 10
        adapter = HTTPAdapter(pool_connections=2, pool_maxsize=pool_size)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _cddis_thread_local.session = session
    return _cddis_thread_local.session


@contextmanager
def ftp_tls(url: str, **kwargs) -> None:
    kwargs.setdefault("timeout", 30)
    with ftplib.FTP_TLS(url, **kwargs) as ftps:
        ftps.login()
        ftps.prot_p()
        yield ftps
        ftps.quit()


def download_atx(download_dir: Path, long_filename: bool = False, if_file_present: str = "prompt_user") -> None:
    """
    Download the ATX file necessary for running the PEA provided the download directory (download_dir)
    """

    if long_filename:
        atx_filename = "igs20.atx"
    else:
        atx_filename = "igs14.atx"
    ensure_folders([download_dir])
    url = f"https://files.igs.org/pub/station/general/{atx_filename}"
    attempt_url_download(
        download_dir=download_dir, url=url, filename=atx_filename, type_of_file="ATX", if_file_present=if_file_present
    )


def download_atmosphere_loading_model(download_dir: Path, if_file_present: str = "prompt_user") -> Path:
    """
    Download the Atmospheric loading BLQ file necessary for running the PPP example
    provided the download directory (download_dir) and what to do if file already present (if_file_present)
    """
    ensure_folders([download_dir])
    download_filepath = attempt_url_download(
        download_dir=download_dir,
        url="https://peanpod.s3.ap-southeast-2.amazonaws.com/aux/products/tables/ALOAD_GO.BLQ.gz",
        filename="ALOAD_GO.BLQ.gz",
        type_of_file="Atmospheric Tide Loading Model",
        if_file_present=if_file_present,
    )

    if download_filepath:
        download_filepath = decompress_file(input_filepath=download_filepath, delete_after_decompression=True)

    return download_filepath


def download_brdc(
    download_dir: Path,
    start_epoch: datetime,
    end_epoch: datetime,
    source: str = "cddis",
    rinexVersion: str = "3",
    filePeriod: str = "01D",
    if_file_present: str = "prompt_user",
) -> None:
    """
    Download the most recent BRDC file/s from CDDIS
    provided the download directory (download_dir)
    """
    # Download Broadcast file/s
    logging.info("Downloading Broadcast files")
    if source.lower() == "gnss-data":
        download_files_from_gnss_data(
            station_list=["BRDC"],
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            data_dir=download_dir,
            file_period=filePeriod,
            file_type="nav",
            rinex_version=rinexVersion,
            if_file_present=if_file_present,
        )
    elif source.lower() == "cddis":
        reference_dt = start_epoch - timedelta(days=1)
        while (end_epoch - reference_dt).total_seconds() > 0:
            doy = reference_dt.strftime("%j")
            brdc_compfile = f"BRDC00IGS_R_{reference_dt.year}{doy}0000_01D_MN.rnx.gz"  # DZ: download MN file
            if check_whether_to_download(
                filename=brdc_compfile, download_dir=download_dir, if_file_present=if_file_present
            ):

                download_file_from_cddis(
                    filename=brdc_compfile,
                    ftp_folder=f"gnss/data/daily/{reference_dt.year}/brdc/",
                    output_folder=download_dir,
                    if_file_present=if_file_present,
                )
            reference_dt += timedelta(days=1)


def download_geomagnetic_model(download_dir: Path, model: str = "igrf14", if_file_present: str = "prompt_user") -> Path:
    """
    Download the International Geomagnetic Reference Field model file necessary for running the PPP example
    provided the download directory (download_dir)
    Default: IGRF14 coefficients
    """
    if model == "igrf14":
        ensure_folders([download_dir])
        download_filepath = attempt_url_download(
            download_dir=download_dir,
            url="https://peanpod.s3.ap-southeast-2.amazonaws.com/aux/products/tables/igrf14coeffs.txt.gz",
            filename="igrf14coeffs.txt.gz",
            type_of_file="Geomagnetic Field coefficients - IGRF14",
            if_file_present=if_file_present,
        )
    else:
        logging.info(f"Unsupported Geomagnetic Field coefficients model type - {model}")
        download_filepath = None

    if download_filepath:
        download_filepath = decompress_file(input_filepath=download_filepath, delete_after_decompression=True)

    return download_filepath


def download_geopotential_model(
    download_dir: Path, model: str = "egm2008", if_file_present: str = "prompt_user"
) -> Path:
    """
    Download the Geopotential model file/s necessary for running the PPP example
    provided the download directory (download_dir)
    Default: EGM2008
    """
    if model == "egm2008":
        ensure_folders([download_dir])
        download_filepath = attempt_url_download(
            download_dir=download_dir,
            url="https://peanpod.s3.ap-southeast-2.amazonaws.com/aux/products/tables/EGM2008.gfc.gz",
            filename="EGM2008.gfc.gz",
            type_of_file="Geopotential Model - EGM2008",
            if_file_present=if_file_present,
        )
    else:
        logging.info(f"Unsupported Geopotential model type - {model}")
        download_filepath = None

    if download_filepath:
        download_filepath = decompress_file(input_filepath=download_filepath, delete_after_decompression=True)

    return download_filepath


def download_ocean_loading_model(download_dir: Path, if_file_present: str = "prompt_user") -> Path:
    """
    Download the Ocean Loading BLQ file necessary for running the PPP example
    provided the download directory (download_dir) and what to do if file already present (if_file_present)
    """
    ensure_folders([download_dir])
    download_filepath = attempt_url_download(
        download_dir=download_dir,
        url="https://peanpod.s3.ap-southeast-2.amazonaws.com/aux/products/tables/OLOAD_GO.BLQ.gz",
        filename="OLOAD_GO.BLQ.gz",
        type_of_file="Ocean Tide Loading Model",
        if_file_present=if_file_present,
    )

    if download_filepath:
        download_filepath = decompress_file(input_filepath=download_filepath, delete_after_decompression=True)

    return download_filepath


def download_ocean_pole_tide_file(download_dir: Path, if_file_present: str = "prompt_user") -> Path:
    """
    Download the Ocean Pole Tide Loading coefficients file necessary for running the PPP example
    provided the download directory (download_dir) and what to do if file already present (if_file_present)
    """
    ensure_folders([download_dir])
    download_filepath = attempt_url_download(
        download_dir=download_dir,
        url="https://peanpod.s3.ap-southeast-2.amazonaws.com/aux/products/tables/opoleloadcoefcmcor.txt.gz",
        filename="opoleloadcoefcmcor.txt.gz",
        type_of_file="Ocean Pole Tide Loading Coefficients",
        if_file_present=if_file_present,
    )

    if download_filepath:
        download_filepath = decompress_file(input_filepath=download_filepath, delete_after_decompression=True)

    return download_filepath


def download_ocean_tide_potential_model(
    download_dir: Path, if_file_present: str = "prompt_user", model: str = "fes2014b"
) -> Path:
    """
    Download the Ocean Tide Potential Model file necessary for running the PPP example
    provided the download directory (download_dir) and what to do if file already present (if_file_present)
    """
    if model == "fes2014b":
        ensure_folders([download_dir])
        download_filepath = attempt_url_download(
            download_dir=download_dir,
            url="https://peanpod.s3.ap-southeast-2.amazonaws.com/aux/products/tables/fes2014b_Cnm-Snm.dat.gz",
            filename="fes2014b_Cnm-Snm.dat.gz",
            type_of_file="Ocean Tide Potential Model - fes2014b",
            if_file_present=if_file_present,
        )
    else:
        logging.info(f"Unsupported Ocean Tide Potential Model type - {model}")
        download_filepath = None

    if download_filepath:
        download_filepath = decompress_file(input_filepath=download_filepath, delete_after_decompression=True)

    return download_filepath


def download_planetary_ephemerides_file(
    download_dir: Path, if_file_present: str = "prompt_user", ephem_file: str = "DE436.1950.2050"
) -> Path:
    """
    Download the Planetary Ephemerides file necessary for running the PPP example
    provided the download directory (download_dir) and what to do if file already present (if_file_present)
    Default: DE436.1950.2050
    """
    if ephem_file == "DE436.1950.2050":
        ensure_folders([download_dir])
        download_filepath = attempt_url_download(
            download_dir=download_dir,
            url="https://peanpod.s3.ap-southeast-2.amazonaws.com/aux/products/tables/DE436.1950.2050.gz",
            filename="DE436.1950.2050.gz",
            type_of_file="Planetary Ephemerides - DE436.1950.2050",
            if_file_present=if_file_present,
        )
    else:
        logging.info(f"Unsupported Planetary Ephemerides type - {ephem_file}")
        download_filepath = None

    if download_filepath:
        download_filepath = decompress_file(input_filepath=download_filepath, delete_after_decompression=True)

    return download_filepath


def download_trop_model(download_dir: Path, if_file_present: str = "prompt_user", model: str = "gpt2") -> Path:
    """
    Download the relevant troposphere model file/s necessary for running the PEA
    provided the download directory (download_dir) and model
    Default is GPT 2.5
    """
    if model == "gpt2":
        ensure_folders([download_dir])
        download_filepath = attempt_url_download(
            download_dir=download_dir,
            url="https://peanpod.s3.ap-southeast-2.amazonaws.com/aux/products/tables/gpt_25.grd.gz",
            filename="gpt_25.grd.gz",
            type_of_file="Troposphere Model",
            if_file_present=if_file_present,
        )
    else:
        logging.info(f"Unsupported Troposphere model type - {model}")
        download_filepath = None

    if download_filepath:
        download_filepath = decompress_file(input_filepath=download_filepath, delete_after_decompression=True)

    return download_filepath


def download_iau2000_file(download_dir: Path, start_epoch: datetime, if_file_present: str = "prompt_user"):
    """
    Download relevant IAU2000 file from CDDIS or IERS based on start_epoch of data
    """
    ensure_folders([download_dir])
    # Download most recent daily IAU2000 file if running for a session within the past week (data is within 3 months)
    if datetime.now() - start_epoch < timedelta(weeks=1):
        url_dir = "daily/"
        iau2000_filename = "finals2000A.daily"
        logging.info("Attempting Download of finals2000A.daily file")
    # Otherwise download the IAU2000 file dating back to 1992
    else:
        url_dir = "standard/"
        iau2000_filename = "finals2000A.data"
        logging.info("Attempting Download of finals2000A.data file")
    # Attempt download from CDDIS first, if that fails try the IERS website
    try:
        logging.info("Downloading IAU2000 file from CDDIS")
        if check_whether_to_download(
            filename="finals.data.iau2000.txt", download_dir=download_dir, if_file_present=if_file_present
        ):
            download_filepath = download_file_from_cddis(
                filename=iau2000_filename,
                ftp_folder="products/iers",
                output_folder=download_dir,
                decompress=False,
                if_file_present=if_file_present,
            )
            (download_dir / iau2000_filename).rename(download_dir / "finals.data.iau2000.txt")
        else:
            return None
    except:
        logging.info("Failed CDDIS download - Downloading IAU2000 file from IERS")
        download_filepath = attempt_url_download(
            download_dir=download_dir,
            url="https://datacenter.iers.org/products/eop/rapid/" + url_dir + iau2000_filename,
            filename="finals.data.iau2000.txt",
            type_of_file="EOP IAU2000",
            if_file_present=if_file_present,
        )
    return download_filepath


def download_satellite_metadata_snx(download_dir: Path, if_file_present: str = "prompt_user") -> Path:
    """
    Download the most recent IGS satellite metadata file
    """
    ensure_folders([download_dir])
    download_filepath = attempt_url_download(
        download_dir=download_dir,
        url="https://files.igs.org/pub/station/general/igs_satellite_metadata.snx",
        filename="igs_satellite_metadata.snx",
        type_of_file="IGS satellite metadata",
        if_file_present=if_file_present,
    )
    return download_filepath


def download_yaw_files(download_dir: Path, if_file_present: str = "prompt_user"):
    """
    Download yaw rate / bias files
    """
    ensure_folders([download_dir])
    urls = [
        "https://peanpod.s3.ap-southeast-2.amazonaws.com/aux/products/tables/bds_yaw_modes.snx.gz",
        "https://peanpod.s3.ap-southeast-2.amazonaws.com/aux/products/tables/qzss_yaw_modes.snx.gz",
        "https://peanpod.s3.ap-southeast-2.amazonaws.com/aux/products/tables/sat_yaw_bias_rate.snx.gz",
    ]
    download_filepaths = []
    targets = ["bds_yaw_modes.snx.gz", "qzss_yaw_modes.snx.gz", "sat_yaw_bias_rate.snx.gz"]

    for url, target in zip(urls, targets):

        download_filepath = attempt_url_download(
            download_dir=download_dir,
            url=url,
            filename=target,
            type_of_file="Yaw Model SNX",
            if_file_present=if_file_present,
        )
        if download_filepath:
            download_filepaths.append(decompress_file(download_filepath, delete_after_decompression=True))

    return download_filepaths


def download_most_recent_cddis_file(
    download_dir: Path,
    pointer_date: GPSDate,
    file_type: str = "SNX",
    long_filename: bool = False,
    analysis_center: str = "IGS",
    if_file_present: str = "prompt_user",
) -> None:
    """
    Download the most recent files from IGS provided the download directory (download_dir) session, and file type
    """
    with ftp_tls("gdc.cddis.eosdis.nasa.gov") as ftps:
        # Move to directory of current week:
        ftps.cwd(f"gnss/products/")
        if pointer_date.gpswk not in ftps.nlst():
            GPS_day = int(pointer_date.gpswkD[-1])
            pointer_date = GPSDate(pointer_date.ts - np.timedelta64(7 + GPS_day, "D"))
        ftps.cwd(f"{pointer_date.gpswk}")
        # Search for most recent file
        logging.info(f"Searching for most recent {file_type} files - Long Filename set to: {long_filename}")
        target_filename, pointer_date, ftps = search_for_most_recent_file(
            pointer_date=pointer_date,
            ftps=ftps,
            long_filename=long_filename,
            file_type="SNX",
            analysis_center=analysis_center,
            timespan=timedelta(days=1),
            solution_type="SNX",
            sampling_rate="01D",
            content_type="CRD",
        )
    # Download recent file:
    download_file_from_cddis(
        filename=target_filename,
        ftp_folder=f"gnss/products/{pointer_date.gpswk}",
        output_folder=download_dir,
        if_file_present=if_file_present,
    )


def search_for_most_recent_file(
    pointer_date: GPSDate,
    ftps: ftplib.FTP_TLS,
    long_filename: bool = False,
    file_type: str = "SNX",
    analysis_center: str = "IGS",
    timespan: timedelta = timedelta(days=7),
    solution_type: str = "SNX",
    sampling_rate: str = "07D",
    content_type: str = None,
    project_type: str = "OPS",
) -> Tuple[str, GPSDate, ftplib.FTP_TLS]:
    """
    Find the most recent file on CDDIS of file type: file_type
    """
    if content_type == None:
        content_type = generate_content_type(file_type, analysis_center)
    # Get list of available files and those of file type: file_type
    most_recent_files_in_dir = ftps.nlst()
    target_filename, pointer_date, _ = generate_product_filename(
        reference_start=pointer_date.as_datetime,
        file_ext=file_type,
        long_filename=long_filename,
        analysis_center=analysis_center,
        timespan=timespan,
        solution_type=solution_type,
        sampling_rate=sampling_rate,
        content_type=content_type,
        project=project_type,
    )
    file_list = [f for f in most_recent_files_in_dir if f == target_filename]
    # If no files of file_type available, keep looking back week by week:
    while file_list == []:
        logging.info(f"GPS week {pointer_date.gpswk} too recent")
        logging.info(f"No IGS {file_type} files found in GPS week {pointer_date.gpswk}")
        logging.info(f"Moving to GPS week {int(pointer_date.gpswk) - 1}")
        # Move pointer_date back and search
        GPS_day = int(pointer_date.gpswkD[-1])
        pointer_date = GPSDate(pointer_date.ts - np.timedelta64(7 + GPS_day, "D"))
        target_filename, _, _ = generate_product_filename(
            reference_start=pointer_date.as_datetime,
            file_ext=file_type,
            long_filename=long_filename,
            analysis_center=analysis_center,
            timespan=timespan,
            solution_type=solution_type,
            sampling_rate=sampling_rate,
            content_type=content_type,
            project=project_type,
        )
        # pointer_date = GPSDate(np.datetime64(reference_epoch))
        logging.info(f"Searching for file: {target_filename}")
        ftps.cwd("../" + pointer_date.gpswk)
        most_recent_files_in_dir = ftps.nlst()
        file_list = [f for f in most_recent_files_in_dir if f == target_filename]
    logging.info("Found IGS file")
    return target_filename, pointer_date, ftps


def download_gnss_data_entry(
    entry: dict, output_dir: Path, max_retries: int = 3, if_file_present: str = "prompt_user"
) -> str:
    file_url = entry["fileLocation"]
    file_type = entry["fileType"]
    retries = 0
    download_done = False
    if file_type == "obs":
        filename = urlparse(file_url).path.split("/")[-1].split(".")[0] + ".rnx"
    elif file_type == "nav":
        filename = ".".join(urlparse(file_url).path.split("/")[-1].split(".")[:2])

    # Try until the file has been downloaded or there's no retries left
    while not download_done and retries <= max_retries:
        try:
            # logging.info(f"Downloading {filename} to {out_path}")
            download_filepath = attempt_url_download(
                download_dir=output_dir,
                url=file_url,
                filename=filename,
                type_of_file="RNX",
                if_file_present=if_file_present,
            )
            download_done = True
            # logging.info(f"Downloaded {filename}")
            return download_filepath
        except:
            retries += 1
            if retries > max_retries:
                logging.warning(f"Failed to download {file_url} and reached maximum retry count ({max_retries}).")

            logging.debug(f"Received an error while try to download {file_url}, retrying({retries}).")
            # Add some backoff time (exponential random as it appears to be contention based?)
            sleep(random.uniform(0.0, 2.0**retries))


def download_files_from_gnss_data(
    station_list: list,
    start_epoch: datetime,
    end_epoch: datetime,
    data_dir: Path,
    file_period: str = "01D",
    rinex_version: int = 3,
    file_type: str = "obs",
    decompress: str = "true",
    if_file_present: str = "prompt_user",
) -> None:
    """Download nav/brdc files from the GA GNSS data API. Not used for obs station files."""
    QUERY_PARAMS = {
        "metadataStatus": "valid",
        "stationId": ",".join(station_list),
        "fileType": file_type,
        "rinexVersion": rinex_version,
        "filePeriod": file_period,
        "decompress": decompress,
        "startDate": start_epoch.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endDate": end_epoch.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tenantId": "default",
    }
    request = requests.get(API_URL + "/rinexFiles", params=QUERY_PARAMS, headers={})
    request.raise_for_status()
    for item in json.loads(request.content):
        download_gnss_data_entry(entry=item, output_dir=data_dir, max_retries=3, if_file_present=if_file_present)


# ── RINEX obs station downloads (GA API + CDDIS fallback) ─────────────────────


def _cddis_rinex_url_folder(date: datetime) -> str:
    """Return the CDDIS url_folder for daily RINEX obs files on the given date."""
    doy = date.timetuple().tm_yday
    return f"gnss/data/daily/{date.year}/{doy:03d}/{date.strftime('%y')}d"


def _cddis_list_date(date: datetime, username: str, password: str, cache_dir: Path) -> list:
    """
    Fetch the CDDIS RINEX3 obs file listing for one day, with disk caching.
    Returns a list of filenames. Only successful responses (including genuine 404s)
    are cached so network errors will be retried on the next run.
    """
    doy = date.timetuple().tm_yday
    cache_file = cache_dir / f"{date.year}{doy:03d}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    url = f"{CDDIS_RINEX_BASE}/{date.year}/{doy:03d}/{date.strftime('%y')}d/*?list"
    session = _get_cddis_session(username, password)

    for attempt in range(3):
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 404:
                result = []
                cache_file.write_text(json.dumps(result))
                return result
            if resp.status_code in (401, 403):
                logging.error(f"CDDIS auth failed ({resp.status_code}) for {date.date()} — check ~/.netrc")
                return []
            resp.raise_for_status()
            filenames = [
                line.split()[0]
                for line in resp.text.strip().split("\n")
                if line.strip() and _RINEX3_OBS_RE.match(line.split()[0])
            ]
            cache_file.write_text(json.dumps(filenames))
            logging.debug(f"CDDIS listing {date.date()}: {len(filenames)} files")
            return filenames
        except requests.Timeout:
            wait = 5 * (2**attempt)
            logging.warning(f"CDDIS listing timeout for {date.date()}, retrying in {wait}s...")
            sleep(wait)
        except requests.RequestException as e:
            logging.error(f"CDDIS listing error for {date.date()}: {e}")
            return []

    logging.error(f"CDDIS: gave up on listing for {date.date()} after 3 attempts")
    return []


def _find_station_in_listing(station_4char: str, filenames: list) -> str:
    """
    Find the best RINEX obs file for a station from a CDDIS directory listing.
    Prefers 30S sampling rate to match GA files. Returns filename or None.
    """
    upper = station_4char.upper()
    matches = [f for f in filenames if f[:4].upper() == upper]
    if not matches:
        return None
    preferred = [f for f in matches if "_30S_MO" in f.upper()]
    return preferred[0] if preferred else matches[0]


def _ga_download_worker(entry: dict, data_dir: Path, if_file_present: str, work_root: Path = None) -> tuple:
    """
    Download one file from a GA API response entry.
    Returns (station_4char, date_str, filepath_or_None).
    Files are downloaded as .crx.gz; use --decompress to decompress after download.
    """
    file_url = entry["fileLocation"]
    raw_name = urlparse(file_url).path.split("/")[-1]
    station_4char = raw_name[:4].upper()
    try:
        epoch_part = raw_name.split("_")[2]  # e.g. "20190010000"
        year, doy = int(epoch_part[:4]), int(epoch_part[4:7])
        date_str = (datetime(year, 1, 1) + timedelta(days=doy - 1)).strftime("%Y-%m-%d")
    except (IndexError, ValueError):
        date_str = "unknown"

    output_dir = _resolve_output_dir(date_str, data_dir, work_root)
    filename = raw_name
    try:
        filepath = attempt_url_download(
            download_dir=output_dir,
            url=file_url,
            filename=filename,
            type_of_file="RINEX obs",
            if_file_present=if_file_present,
        )
        return station_4char, date_str, filepath
    except Exception as e:
        logging.warning(f"GA download failed for {filename}: {e}")
        return station_4char, date_str, None


def _cddis_download_worker(
    filename: str,
    url_folder: str,
    data_dir: Path,
    station_4char: str,
    date_str: str,
    username: str,
    password: str,
    if_file_present: str = "prompt_user",
    work_root: Path = None,
) -> tuple:
    """
    Download one RINEX file from CDDIS as .crx.gz (decompression handled separately).
    Uses a thread-local session to reuse connections and avoid per-file auth overhead.
    Returns (station_4char, date_str, filepath_or_None).
    """
    output_dir = _resolve_output_dir(date_str, data_dir, work_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = _get_cddis_session(username, password)
    try:
        result = download_file_from_cddis(
            filename=filename,
            url_folder=url_folder,
            output_folder=output_dir,
            max_retries=3,
            decompress=False,
            if_file_present=if_file_present,
            session=session,
        )
    except Exception as e:
        logging.error(f"CDDIS: gave up on {filename}: {e}")
        return station_4char, date_str, None
    return station_4char, date_str, result


def _resolve_output_dir(date_str: str, data_dir: Path, work_root: Path) -> Path:
    """Return the output directory for a given date, creating it if needed."""
    if work_root:
        d = work_root / date_str / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return data_dir


def _scan_existing_files(stations: list, start_epoch: datetime, end_epoch: datetime, data_dir: Path = None, work_root: Path = None) -> set:
    """
    Scan data_dir for already-present RINEX files (.crx.gz or .rnx).
    Returns a set of (station_4char, date_str) pairs that already have a file on disk.
    """
    existing = set()
    current = start_epoch.replace(hour=0, minute=0, second=0, microsecond=0)
    end = end_epoch.replace(hour=0, minute=0, second=0, microsecond=0)
    station_set = {s.upper() for s in stations}
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        scan_dir = _resolve_output_dir(date_str, data_dir, work_root)
        if scan_dir.exists():
            for pattern in ("*.crx.gz", "*.rnx"):
                for f in scan_dir.glob(pattern):
                    code = f.name[:4].upper()
                    if code in station_set:
                        existing.add((code, date_str))
        current += timedelta(days=1)
    return existing


def _write_provenance_log(provenance: list, log_path: Path) -> None:
    """Append (station, date, filename, source) records to a CSV provenance log."""
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["station", "date", "filename", "source"])
        for station, date_str, filepath, source in provenance:
            writer.writerow([station, date_str, filepath.name if filepath else "", source])


def _download_rinex_from_ga(
    station_list: list,
    start_epoch: datetime,
    end_epoch: datetime,
    data_dir: Path,
    file_period: str,
    rinex_version: int,
    if_file_present: str,
    max_workers: int,
    work_root: Path = None,
) -> tuple:
    """
    Query the GA API and download all available obs files in parallel.
    Does not filter by metadataStatus so files with invalid metadata are included.
    Returns (ga_downloaded: set of (station, date_str), provenance: list).
    """
    # Query one day + 10 stations at a time to avoid gateway timeouts
    GA_STATION_CHUNK = 10
    entries = []
    current = start_epoch.replace(hour=0, minute=0, second=0, microsecond=0)
    end = end_epoch.replace(hour=0, minute=0, second=0, microsecond=0)
    num_days = (end - current).days + 1
    station_chunks = [station_list[i:i + GA_STATION_CHUNK] for i in range(0, len(station_list), GA_STATION_CHUNK)]
    logging.info(f"GA: Querying API for {len(station_list)} stations over {num_days} day(s) in {len(station_chunks)} station chunk(s)")
    while current <= end:
        next_day = current + timedelta(days=1)
        for chunk in station_chunks:
            params = {
                "stationId": ",".join(chunk),
                "fileType": "obs",
                "rinexVersion": rinex_version,
                "filePeriod": file_period,
                "decompress": "false",
                "startDate": current.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endDate": next_day.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tenantId": "default",
            }
            try:
                resp = requests.get(API_URL + "/rinexFiles", params=params, timeout=60)
                resp.raise_for_status()
                chunk_entries = json.loads(resp.content)
                entries.extend(chunk_entries)
                logging.debug(f"GA: {current.date()} ({chunk[0]}–{chunk[-1]}): {len(chunk_entries)} files")
            except requests.RequestException as e:
                logging.warning(f"GA API query failed for {current.date()}, stations {chunk[0]}–{chunk[-1]}: {e}")
        current = next_day

    logging.info(f"GA: {len(entries)} files available, downloading with {max_workers} workers")
    ga_downloaded = set()
    provenance = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_ga_download_worker, entry, data_dir, if_file_present, work_root): entry for entry in entries}
        for future in as_completed(futures):
            station, date_str, filepath = future.result()
            if filepath:
                ga_downloaded.add((station, date_str))
                provenance.append((station, date_str, filepath, "ga"))
    logging.info(f"GA: {len(ga_downloaded)} files downloaded")
    return ga_downloaded, provenance


def _prefetch_cddis_listings(
    missing_dates: list, username: str, password: str, cache_dir: Path, max_workers: int
) -> dict:
    """
    Concurrently fetch CDDIS directory listings for each date that still has missing stations.
    Results are cached to disk so reruns only fetch what's not already cached.
    Returns dict mapping date_str -> list of available filenames.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_listings = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(missing_dates))) as executor:
        futures = {
            executor.submit(_cddis_list_date, datetime.strptime(d, "%Y-%m-%d"), username, password, cache_dir): d
            for d in missing_dates
        }
        for future in as_completed(futures):
            date_str = futures[future]
            all_listings[date_str] = future.result()
    logging.info(f"CDDIS: Listings fetched for {len(missing_dates)} dates")
    return all_listings


def _download_rinex_from_cddis(
    missing_pairs: set, all_listings: dict, data_dir: Path, username: str, password: str, max_workers: int, if_file_present: str = "prompt_user", work_root: Path = None
) -> list:
    """
    Build download tasks from the CDDIS listings (skipping stations not in any listing
    to avoid wasted requests), then download in a single parallel pool.
    Returns provenance list of (station, date_str, filepath, 'cddis').
    """
    tasks = []
    not_found = 0
    for station, date_str in missing_pairs:
        listing = all_listings.get(date_str, [])
        if not listing:
            continue
        fname = _find_station_in_listing(station, listing)
        if fname:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            tasks.append((fname, _cddis_rinex_url_folder(date), station, date_str))
        else:
            not_found += 1

    if not_found:
        logging.debug(f"CDDIS: {not_found} station-days not in any listing")
    logging.info(f"CDDIS: Downloading {len(tasks)} files with {max_workers} workers")

    provenance = []
    cddis_ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_cddis_download_worker, fname, url_folder, data_dir, station, date_str, username, password, if_file_present, work_root): (
                station,
                date_str,
            )
            for fname, url_folder, station, date_str in tasks
        }
        for future in as_completed(futures):
            station, date_str, filepath = future.result()
            if filepath:
                cddis_ok += 1
                provenance.append((station, date_str, filepath, "cddis"))
    logging.info(f"CDDIS: {cddis_ok}/{len(tasks)} files downloaded")
    return provenance


def download_rinex_obs(
    station_list: list,
    start_epoch: datetime,
    end_epoch: datetime,
    data_dir: Path,
    source: str = "ga",
    file_period: str = "01D",
    rinex_version: int = 3,
    if_file_present: str = "prompt_user",
    max_workers: int = 8,
    decompress: bool = False,
    delete_compressed: bool = False,
    dated_dirs: bool = False,
) -> None:
    """
    Download RINEX obs files for a list of stations from GA API and/or CDDIS.

    source: 'ga'    - GA GNSS data API only (includes files with invalid metadata)
            'cddis' - CDDIS only (requires NASA Earthdata credentials in ~/.netrc)
            'both'  - GA first, CDDIS fallback for any missing station-days

    For 'both': GA downloads run first, then CDDIS listings are prefetched concurrently
    only for dates with gaps, then missing files are downloaded in one pool.
    Writes a provenance log to {data_dir}/rinex_provenance.csv.
    """
    ensure_folders([data_dir])
    stations_upper = [s.upper() for s in station_list]
    provenance = []
    work_root = data_dir if dated_dirs else None

    # Pre-scan disk for files from previous runs so dont-replace skips are accounted for
    already_on_disk = _scan_existing_files(stations_upper, start_epoch, end_epoch, data_dir, work_root)

    # Phase 1: GA
    ga_downloaded = set()
    if source in ("ga", "both"):
        ga_downloaded, ga_provenance = _download_rinex_from_ga(
            stations_upper, start_epoch, end_epoch, data_dir, file_period, rinex_version, if_file_present, max_workers, work_root
        )
        provenance.extend(ga_provenance)
        ga_downloaded |= already_on_disk

        if source == "ga":
            current = start_epoch.replace(hour=0, minute=0, second=0, microsecond=0)
            end = end_epoch.replace(hour=0, minute=0, second=0, microsecond=0)
            all_pairs = {
                (s, (current + timedelta(days=i)).strftime("%Y-%m-%d"))
                for s in stations_upper
                for i in range((end - current).days + 1)
            }
            missing = all_pairs - ga_downloaded
            if missing:
                missing_stations = sorted({s for s, _ in missing})
                logging.warning(
                    f"GA: {len(missing)} station-days not downloaded "
                    f"({len(missing_stations)} stations). "
                    f"Consider --rinex-source both to fall back to CDDIS for missing files."
                )

    # Phase 2: CDDIS for any missing station-days
    if source in ("cddis", "both"):
        try:
            username, password = get_earthdata_credentials()
        except Exception as e:
            logging.error(f"CDDIS: Cannot load Earthdata credentials: {e}")
            logging.error("Add to ~/.netrc: machine urs.earthdata.nasa.gov login <user> password <pass>")
            if source == "cddis":
                return
            logging.warning("Skipping CDDIS fallback")
            username = None

        if username:
            current = start_epoch.replace(hour=0, minute=0, second=0, microsecond=0)
            end = end_epoch.replace(hour=0, minute=0, second=0, microsecond=0)
            all_pairs = {
                (s, (current + timedelta(days=i)).strftime("%Y-%m-%d"))
                for s in stations_upper
                for i in range((end - current).days + 1)
            }
            missing_pairs = all_pairs - ga_downloaded - already_on_disk

            if not missing_pairs:
                if source == "both":
                    logging.info("CDDIS: GA had complete coverage, no CDDIS downloads needed")
                return
            else:
                if source == "both":
                    logging.info(f"CDDIS: {len(missing_pairs)} station-days missing from GA")
                missing_dates = sorted({d for _, d in missing_pairs})
                cache_dir = (work_root if work_root else data_dir) / ".cddis_listings"
                all_listings = _prefetch_cddis_listings(missing_dates, username, password, cache_dir, max_workers)
                cddis_provenance = _download_rinex_from_cddis(
                    missing_pairs, all_listings, data_dir, username, password, max_workers, if_file_present, work_root
                )
                provenance.extend(cddis_provenance)

    # Decompress .crx.gz files if requested
    if decompress:
        search_root = work_root if work_root else data_dir
        gz_files = list(search_root.rglob("*.crx.gz"))
        if gz_files:
            logging.info(f"Decompressing {len(gz_files)} .crx.gz files")
            for gz_file in gz_files:
                try:
                    decompress_file(gz_file, delete_after_decompression=delete_compressed)
                except Exception as e:
                    logging.warning(f"Failed to decompress {gz_file.name}: {e}")

    # Write provenance log
    if provenance:
        log_root = work_root if work_root else data_dir
        log_path = log_root / "rinex_provenance.csv"
        _write_provenance_log(provenance, log_path)
        logging.info(f"Provenance log: {log_path} ({len(provenance)} entries)")
    else:
        logging.warning(f"No RINEX obs files downloaded for any of the {len(stations_upper)} requested stations")


def most_recent_6_hour():
    """
    Returns a datetime object set to the most recent hour divisible by 6: (0000, 0600, 1200 or 1800)
    """
    now_time = datetime.now()
    now_hour = now_time.hour
    # Subtract the remainder of division by 6, to get the most recent hour evenly divisible by 6.
    # E.g. for hour 17: 17 % 6 = 5. 17-5=12.
    latest_hour_interval = now_hour - (now_hour % 6)
    return now_time.replace(hour=latest_hour_interval, minute=0, second=0, microsecond=0)


def auto_download(
    target_dir: Path,
    preset: str,
    station_list: str,
    start_datetime: str,
    end_datetime: str,
    replace: bool,
    dont_replace: bool,
    most_recent: bool,
    analysis_center: str,
    atx: bool,
    aload: bool,
    igrf: bool,
    egm: bool,
    oload: bool,
    opole: bool,
    fes: bool,
    planet: bool,
    sat_meta: bool,
    yaw: bool,
    snx: bool,
    nav: bool,
    sp3: bool,
    erp: bool,
    clk: bool,
    bia: bool,
    gpt2: bool,
    rinex_data_dir: Path,
    trop_dir: Path,
    model_dir: Path,
    solution_type: str,
    project_type: str,
    rinex_file_period: str,
    bia_ac: str,
    iau2000: bool,
    datetime_format: str,
    data_source: str,
    campaign: str,
    product_version: str,
    rinex_source: str,
    rinex_workers: int,
    decompress: bool,
    delete_compressed: bool,
    rinex_dated_dirs: bool,
    verbose: bool,
) -> None:
    configure_logging(verbose)

    # Assign flags for preset selection
    if preset == "real-time":
        most_recent = True
        atx = True
        oload = True
        gpt2 = True
        snx = True
        sat_meta = True
        yaw = True

    if preset == "igs-station":
        atx = True
        oload = True
        aload = True
        igrf = True
        opole = True
        planet = True
        gpt2 = True
        snx = True
        sat_meta = True
        yaw = True
        nav = True
        sp3 = True
        erp = True
        clk = True
        bia = True

    if solution_type in ["FIN", "RAP"]:
        timespan = timedelta(days=1)
    elif solution_type == "ULT":
        timespan = timedelta(days=2)

    # Assign start and end datetimes (if provided)
    if start_datetime:
        start_epoch = datetime.strptime(start_datetime, datetime_format)
        start_gpsdate = GPSDate(start_epoch.strftime("%Y-%m-%dT%H:%M"))
        long_filename = long_filename_cddis_cutoff(epoch=start_epoch)
    else:
        start_epoch = datetime.now()
        start_gpsdate = GPSDate("today")
        long_filename = long_filename_cddis_cutoff(epoch=datetime.today())

    if end_datetime:
        end_epoch = datetime.strptime(end_datetime, datetime_format)

    # If directories haven't been assigned use default: target-dir
    if not rinex_data_dir:
        rinex_data_dir = target_dir
    if not trop_dir:
        trop_dir = target_dir / "tables"
    if not model_dir:
        model_dir = target_dir / "tables"
    if not rinex_file_period:
        rinex_file_period = "01D"

    # Determine what to do when file already present in target-dir:
    if replace and dont_replace:
        raise Exception("Cannot set both --replace and --dont-replace flags")
    elif replace:
        if_file_present = "replace"
    elif dont_replace:
        if_file_present = "dont_replace"
    else:
        if_file_present = "prompt_user"

    # Ensure the directories exist:
    dir_list = [target_dir, rinex_data_dir, trop_dir, model_dir]
    ensure_folders(dir_list)

    # Assign variables based on flags
    if gpt2:
        trop_model = "gpt2"
    else:
        trop_model = None
    if most_recent:
        start_gpsdate = GPSDate("today")
        start_epoch = most_recent_6_hour()
        end_epoch = datetime.now()
        long_filename = long_filename_cddis_cutoff(epoch=datetime.today())

    tasks = []
    max_workers = os.cpu_count() or 4

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if atx:
            tasks.append(executor.submit(
                download_atx,
                download_dir=target_dir,
                long_filename=long_filename,
                if_file_present=if_file_present,
            ))

        if oload:
            tasks.append(executor.submit(
                download_ocean_loading_model,
                download_dir=model_dir,
                if_file_present=if_file_present,
            ))

        if aload:
            tasks.append(executor.submit(
                download_atmosphere_loading_model,
                download_dir=model_dir,
                if_file_present=if_file_present,
            ))

        if igrf:
            tasks.append(executor.submit(
                download_geomagnetic_model,
                download_dir=model_dir,
                if_file_present=if_file_present,
            ))

        if egm:
            tasks.append(executor.submit(
                download_geopotential_model,
                download_dir=model_dir,
                if_file_present=if_file_present,
            ))

        if opole:
            tasks.append(executor.submit(
                download_ocean_pole_tide_file,
                download_dir=model_dir,
                if_file_present=if_file_present,
            ))

        if fes:
            tasks.append(executor.submit(
                download_ocean_tide_potential_model,
                download_dir=model_dir,
                if_file_present=if_file_present,
            ))

        if planet:
            tasks.append(executor.submit(
                download_planetary_ephemerides_file,
                download_dir=model_dir,
                if_file_present=if_file_present,
            ))

        if trop_model:
            tasks.append(executor.submit(
                download_trop_model,
                download_dir=trop_dir,
                model=trop_model,
                if_file_present=if_file_present,
            ))

        if sat_meta:
            tasks.append(executor.submit(
                download_satellite_metadata_snx,
                download_dir=target_dir,
                if_file_present=if_file_present,
            ))

        if yaw:
            tasks.append(executor.submit(
                download_yaw_files,
                download_dir=model_dir,
                if_file_present=if_file_present,
            ))

        if nav:
            tasks.append(executor.submit(
                download_brdc,
                download_dir=target_dir,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                source=data_source,
                filePeriod=rinex_file_period,
                if_file_present=if_file_present,
            ))

        if snx:
            def snx_task():
                try:
                    if most_recent:
                        return download_most_recent_cddis_file(
                            download_dir=target_dir,
                            pointer_date=start_gpsdate,
                            file_type="SNX",
                            long_filename=long_filename,
                            analysis_center="IGS",
                            if_file_present=if_file_present,
                        )
                    return download_product_from_cddis(
                        download_dir=target_dir,
                        start_epoch=start_epoch,
                        end_epoch=end_epoch,
                        file_ext="SNX",
                        limit=None,
                        long_filename=long_filename,
                        analysis_center="IGS",
                        solution_type="SNX",
                        sampling_rate=generate_sampling_rate(
                            file_ext="SNX", analysis_center="IGS", solution_type="SNX"
                        ),
                        campaign=campaign,
                        version=product_version if product_version is not None else "0",
                        timespan=timedelta(days=1),
                        if_file_present=if_file_present,
                    )
                except Exception as e:
                    logging.info(f"SNX error {e}, trying most recent")
                    return download_most_recent_cddis_file(
                        download_dir=target_dir,
                        pointer_date=start_gpsdate,
                        file_type="SNX",
                        long_filename=long_filename,
                        analysis_center="IGS",
                        if_file_present=if_file_present,
                    )
            tasks.append(executor.submit(snx_task))

        if sp3:
            tasks.append(executor.submit(
                download_product_from_cddis,
                download_dir=target_dir,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                file_ext="SP3",
                limit=None,
                long_filename=long_filename,
                analysis_center=analysis_center,
                solution_type=solution_type,
                project_type=project_type,
                campaign=campaign,
                version=product_version if product_version is not None else "0",
                sampling_rate="05M",
                timespan=timespan,
                if_file_present=if_file_present,
            ))

        if erp:
            if iau2000:
                tasks.append(executor.submit(
                    download_iau2000_file,
                    download_dir=target_dir,
                    start_epoch=start_epoch,
                    if_file_present=if_file_present,
                ))
            else:
                tasks.append(executor.submit(
                    download_product_from_cddis,
                    download_dir=target_dir,
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                    file_ext="ERP",
                    limit=None,
                    long_filename=long_filename,
                    analysis_center=analysis_center,
                    solution_type=solution_type,
                    project_type=project_type,
                    campaign=campaign,
                    version=product_version if product_version is not None else "0",
                    sampling_rate=generate_sampling_rate(
                        file_ext="ERP", analysis_center=analysis_center, solution_type=solution_type
                    ),
                    timespan=timespan,
                    if_file_present=if_file_present,
                ))

        if clk:
            tasks.append(executor.submit(
                download_product_from_cddis,
                download_dir=target_dir,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                file_ext="CLK",
                limit=None,
                long_filename=long_filename,
                analysis_center=analysis_center,
                solution_type=solution_type,
                project_type=project_type,
                campaign=campaign,
                version=product_version if product_version is not None else "0",
                sampling_rate=generate_sampling_rate(
                    file_ext="CLK", analysis_center=analysis_center, solution_type=solution_type
                ),
                timespan=timespan,
                if_file_present=if_file_present,
            ))

        if bia:
            tasks.append(executor.submit(
                download_product_from_cddis,
                download_dir=target_dir,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                file_ext="BIA",
                limit=None,
                long_filename=long_filename,
                analysis_center=bia_ac,
                solution_type=solution_type,
                project_type=project_type,
                campaign=campaign,
                version=product_version if product_version is not None else "0",
                sampling_rate=generate_sampling_rate(
                    file_ext="BIA", analysis_center=analysis_center, solution_type=solution_type
                ),
                timespan=timespan,
                if_file_present=if_file_present,
            ))

        if station_list:
            tasks.append(executor.submit(
                download_rinex_obs,
                station_list=station_list,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                data_dir=rinex_data_dir,
                source=rinex_source,
                file_period=rinex_file_period,
                if_file_present=if_file_present,
                max_workers=rinex_workers,
                decompress=decompress,
                delete_compressed=delete_compressed,
                dated_dirs=rinex_dated_dirs,
            ))

        # Wait for all
        for future in as_completed(tasks):
            try:
                result = future.result()
            except Exception as e:
                logging.error(f"Download failed: {e}")
                raise e


@click.command()
@click.option("--target-dir", required=True, help="Directory to place file downloads", type=Path)
@click.option(
    "--preset",
    help="""Choose from:
    \n\n-'manual' (choose individual flags to download desired files),
    \n\n-'real-time' (download files for processing real-time streams),
    \n\n-'igs-station' (download files for processing IGS CORS stations - include start/end-datetime)
    \n\nDefault: manual""",
    default="manual",
    type=str,
)
@click.option(
    "--station-list",
    help="Provide comma-separated list of IGS stations to download - daily observation RNX files",
    type=str,
)
@click.option(
    "--station-list-file",
    help="Read file of newline separated list of IGS stations to download - takes precedence over option --station-list",
    type=click.File("r"),
)
@click.option("--start-datetime", help="Start of date-time period to download files for", type=str)
@click.option("--end-datetime", help="End of date-time period to download files for", type=str)
@click.option("--replace", help=" Re-download all files already present in target-dir", default=False, is_flag=True)
@click.option("--dont-replace", help="Skip all files already present in target-dir", default=False, is_flag=True)
@click.option("--most-recent", help="Set to download latest version of files", default=False, is_flag=True)
@click.option("--analysis-center", help="Analysis center of files to download", default="IGS", type=str)
@click.option("--atx", help="Flag to Download ATX file", default=False, is_flag=True)
@click.option("--aload", help="Flag to Download Atmospheric Loading file", default=False, is_flag=True)
@click.option("--igrf", help="Flag to Download IGRF14 file", default=False, is_flag=True)
@click.option("--egm", help="Flag to Download EGM2008 file", default=False, is_flag=True)
@click.option("--oload", help="Flag to Download Ocean Tide Loading file", default=False, is_flag=True)
@click.option("--opole", help="Flag to Download Ocean Pole Tide Coefficients", default=False, is_flag=True)
@click.option("--fes", help="Flag to Download FES2014b Ocean Potential File", default=False, is_flag=True)
@click.option("--planet", help="Flag to Download Planetary Ephemerides: DE436.1950.2050", default=False, is_flag=True)
@click.option("--sat-meta", help="Flag to Download Satellite Metadata SNX", default=False, is_flag=True)
@click.option("--yaw", help="Flag to Download Satellite Yaw files", default=False, is_flag=True)
@click.option("--snx", help="Flag to Download Station Position SNX / SSC file", default=False, is_flag=True)
@click.option("--nav", help="Flag to Download navigation / broadcast file/s", default=False, is_flag=True)
@click.option("--sp3", help="Flag to Download SP3 file/s", default=False, is_flag=True)
@click.option("--erp", help="Flag to Download ERP file/s", default=False, is_flag=True)
@click.option("--clk", help="Flag to Download CLK file/s", default=False, is_flag=True)
@click.option("--bia", help="Flag to Download BIA bias file", default=False, is_flag=True)
@click.option("--gpt2", help="Flag to Download GPT 2.5 file", default=False, is_flag=True)
@click.option("--rinex-data-dir", help="Directory to Download RINEX data file/s. Default: target-dir", type=Path)
@click.option("--trop-dir", help="Directory to Download troposphere model file/s. Default: target-dir", type=Path)
@click.option("--model-dir", help="Directory to Download static model files. Default: target-dir / tables", type=Path)
@click.option(
    "--solution-type",
    help="The solution type of products to download from CDDIS. 'FIN': final, or 'RAP': rapid, or 'ULT': ultra-rapid. Default: RAP",
    default="RAP",
    type=str,
)
@click.option(
    "--project-type",
    help="The project type of products to download from CDDIS. 'OPS', 'MGX', 'EXP'. Default: OPS",
    default="OPS",
    type=str,
)
@click.option(
    "--rinex-file-period",
    help="File period of RINEX files to download, e.g. 01D, 01H, 15M. Default: 01D",
    default="01D",
    type=str,
)
@click.option("--bia-ac", help="Analysis center of BIA files to download. Default: COD", default="COD", type=str)
@click.option(
    "--datetime-format",
    help="Format of input datetime string. Default: %Y-%m-%d_%H:%M:%S",
    default="%Y-%m-%d_%H:%M:%S",
    type=str,
)
@click.option("--iau2000", help="Flag to download IAU2000 file for ERP. Default: True", default=True, type=bool)
@click.option(
    "--datetime-format",
    help="Format of input datetime string. Default: %Y-%m-%d_%H:%M:%S",
    default="%Y-%m-%d_%H:%M:%S",
    type=str,
)
@click.option(
    "--data-source",
    help="Source of data for Broadcast files: CDDIS or gnss-data. Default: gnss-data",
    default="gnss-data",
    type=str,
)
@click.option(
    "--campaign",
    help="IGS reprocessing campaign: 'repro1', 'repro2', or 'repro3' (repro3 valid for GPS weeks 729-2237). Default: None (standard products)",
    default=None,
    type=click.Choice(["repro1", "repro2", "repro3"], case_sensitive=False),
)
@click.option(
    "--product-version",
    help="Product version identifier (e.g., '0', '1', '2'). Default: None (auto-determined based on campaign and file type)",
    default=None,
    type=str,
)
@click.option(
    "--rinex-source",
    help="Source for RINEX obs files: 'ga' (GA API, includes invalid metadata), 'cddis', or 'both' (GA preferred, CDDIS fallback). Default: ga",
    default="ga",
    type=click.Choice(["ga", "cddis", "both"], case_sensitive=False),
)
@click.option(
    "--rinex-workers",
    help="Number of parallel download workers for RINEX obs files. Default: 8",
    default=8,
    type=int,
)
@click.option("--decompress", is_flag=True, help="Decompress .crx.gz RINEX files after download")
@click.option("--delete-compressed", is_flag=True, help="Delete .crx.gz files after decompressing (requires --decompress)")
@click.option("--rinex-dated-dirs", is_flag=True, help="Download RINEX obs files into dated subdirectories: {rinex-data-dir}/YYYY-MM-DD/data/")
@click.option("--verbose", is_flag=True)
def auto_download_main(
    target_dir,
    preset,
    station_list,
    station_list_file,
    start_datetime,
    end_datetime,
    replace,
    dont_replace,
    most_recent,
    analysis_center,
    atx,
    aload,
    igrf,
    egm,
    oload,
    opole,
    fes,
    planet,
    sat_meta,
    yaw,
    snx,
    nav,
    sp3,
    erp,
    clk,
    bia,
    gpt2,
    rinex_data_dir,
    trop_dir,
    model_dir,
    solution_type,
    project_type,
    rinex_file_period,
    bia_ac,
    iau2000,
    datetime_format,
    data_source,
    campaign,
    product_version,
    rinex_source,
    rinex_workers,
    decompress,
    delete_compressed,
    rinex_dated_dirs,
    verbose,
):
    try:
        station_list
    except NameError:
        station_list = None
    if not station_list == None:
        station_list = station_list.split(",")
    if station_list_file:
        content = station_list_file.read()
        station_list = content.split("\n")
    auto_download(
        target_dir,
        preset,
        station_list,
        start_datetime,
        end_datetime,
        replace,
        dont_replace,
        most_recent,
        analysis_center,
        atx,
        aload,
        igrf,
        egm,
        oload,
        opole,
        fes,
        planet,
        sat_meta,
        yaw,
        snx,
        nav,
        sp3,
        erp,
        clk,
        bia,
        gpt2,
        rinex_data_dir,
        trop_dir,
        model_dir,
        solution_type,
        project_type,  # DZ: add project_type option
        rinex_file_period,
        bia_ac,
        iau2000,
        datetime_format,
        data_source,
        campaign,
        product_version,
        rinex_source,
        rinex_workers,
        decompress,
        delete_compressed,
        rinex_dated_dirs,
        verbose,
    )


if __name__ == "__main__":
    auto_download_main()
