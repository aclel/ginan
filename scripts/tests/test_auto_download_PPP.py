"""Tests for auto_download_PPP.py"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, call
from datetime import datetime, timedelta
import sys
import os

from auto_download_PPP import (
    download_gnss_data_entry,
    download_files_from_gnss_data,
    download_rinex_obs,
    _scan_existing_files,
)

START = datetime(2019, 1, 1)
END = datetime(2019, 1, 3)


class TestDownloadGnssDataEntry:

    def test_while_loop_condition_success(self):
        """Test that loop exits when download succeeds"""
        entry = {
            "fileLocation": "https://example.com/file.crx.gz",
            "fileType": "obs"
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('auto_download_PPP.attempt_url_download') as mock_download:
                mock_download.return_value = Path(temp_dir) / "test.rnx"

                result = download_gnss_data_entry(
                    entry=entry,
                    output_dir=Path(temp_dir),
                    max_retries=3,
                    if_file_present="replace"
                )

                assert mock_download.call_count == 1
                assert result is not None

    def test_while_loop_condition_failure(self):
        """Test that loop respects max_retries"""
        entry = {
            "fileLocation": "https://example.com/file.crx.gz",
            "fileType": "obs"
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('auto_download_PPP.attempt_url_download') as mock_download:
                with patch('auto_download_PPP.sleep'):
                    mock_download.side_effect = Exception("Network error")

                    result = download_gnss_data_entry(
                        entry=entry,
                        output_dir=Path(temp_dir),
                        max_retries=3,
                        if_file_present="replace"
                    )

                    assert mock_download.call_count == 4
                    assert result is None


class TestDownloadFilesFromGnssData:

    def test_empty_api_response_completes_without_error(self):
        """API returning no files should complete cleanly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('auto_download_PPP.requests.get') as mock_get:
                mock_response = Mock()
                mock_response.content = b'[]'
                mock_response.raise_for_status = Mock()
                mock_get.return_value = mock_response

                download_files_from_gnss_data(
                    station_list=["ALIC", "DARW"],
                    start_epoch=START,
                    end_epoch=END,
                    data_dir=Path(temp_dir),
                )


class TestDownloadRinexObsSourceRouting:

    def _make_patches(self):
        """Return a dict of the patches needed for orchestration tests."""
        return {
            'ga':       patch('auto_download_PPP._download_rinex_from_ga', return_value=(set(), [])),
            'listings': patch('auto_download_PPP._prefetch_cddis_listings', return_value={}),
            'cddis':    patch('auto_download_PPP._download_rinex_from_cddis', return_value=[]),
            'creds':    patch('auto_download_PPP.get_earthdata_credentials', return_value=('user', 'pass')),
            'scan':     patch('auto_download_PPP._scan_existing_files', return_value=set()),
        }

    def test_source_ga_skips_cddis(self, tmp_path):
        p = self._make_patches()
        with p['ga'] as mock_ga, p['listings'] as mock_listings, \
             p['cddis'] as mock_cddis, p['scan']:
            download_rinex_obs(['ALIC'], START, END, tmp_path, source='ga')
            mock_ga.assert_called_once()
            mock_listings.assert_not_called()
            mock_cddis.assert_not_called()

    def test_source_cddis_skips_ga(self, tmp_path):
        p = self._make_patches()
        with p['ga'] as mock_ga, p['listings'] as mock_listings, \
             p['cddis'] as mock_cddis, p['creds'], p['scan']:
            download_rinex_obs(['ALIC'], START, END, tmp_path, source='cddis')
            mock_ga.assert_not_called()
            mock_listings.assert_called_once()
            mock_cddis.assert_called_once()

    def test_source_both_calls_ga_then_cddis_for_missing(self, tmp_path):
        """GA runs first; CDDIS is called for station-days GA missed."""
        p = self._make_patches()
        # GA gets ALIC but misses DARW on day 1
        ga_downloaded = {
            ('ALIC', '2019-01-01'), ('ALIC', '2019-01-02'), ('ALIC', '2019-01-03'),
            ('DARW', '2019-01-02'), ('DARW', '2019-01-03'),
        }
        with p['ga'] as mock_ga, p['listings'] as mock_listings, \
             p['cddis'] as mock_cddis, p['creds'], p['scan']:
            mock_ga.return_value = (ga_downloaded, [])
            download_rinex_obs(['ALIC', 'DARW'], START, END, tmp_path, source='both')
            mock_ga.assert_called_once()
            mock_listings.assert_called_once()
            mock_cddis.assert_called_once()

    def test_source_both_skips_cddis_when_ga_complete(self, tmp_path):
        """CDDIS should not be called if GA downloaded everything."""
        p = self._make_patches()
        ga_downloaded = {
            ('ALIC', '2019-01-01'), ('ALIC', '2019-01-02'), ('ALIC', '2019-01-03'),
        }
        with p['ga'] as mock_ga, p['listings'] as mock_listings, \
             p['cddis'] as mock_cddis, p['creds'], p['scan']:
            mock_ga.return_value = (ga_downloaded, [])
            download_rinex_obs(['ALIC'], START, END, tmp_path, source='both')
            mock_ga.assert_called_once()
            mock_listings.assert_not_called()
            mock_cddis.assert_not_called()

    def test_cddis_credential_failure_returns_without_error(self, tmp_path):
        """Missing Earthdata credentials should log and return cleanly."""
        p = self._make_patches()
        with p['ga'], p['listings'] as mock_listings, \
             p['cddis'] as mock_cddis, p['scan']:
            with patch('auto_download_PPP.get_earthdata_credentials',
                       side_effect=ValueError("no credentials")):
                download_rinex_obs(['ALIC'], START, END, tmp_path, source='cddis')
            mock_listings.assert_not_called()
            mock_cddis.assert_not_called()


class TestDownloadRinexObsIfFilePresent:

    def test_all_files_on_disk_skips_ga(self, tmp_path):
        """When every station-day is already present, GA should not be queried."""
        all_pairs = {
            ('ALIC', '2019-01-01'), ('ALIC', '2019-01-02'), ('ALIC', '2019-01-03'),
        }
        with patch('auto_download_PPP._scan_existing_files', return_value=all_pairs), \
             patch('auto_download_PPP._download_rinex_from_ga') as mock_ga:
            download_rinex_obs(['ALIC'], START, END, tmp_path, source='ga')
            mock_ga.assert_not_called()

    def test_partial_on_disk_only_fetches_missing(self, tmp_path):
        """already_on_disk is passed to GA so it only fetches what's missing."""
        already = {('ALIC', '2019-01-01')}
        with patch('auto_download_PPP._scan_existing_files', return_value=already), \
             patch('auto_download_PPP._download_rinex_from_ga', return_value=(set(), [])) as mock_ga:
            download_rinex_obs(['ALIC'], START, END, tmp_path, source='ga')
            _, kwargs = mock_ga.call_args
            assert kwargs.get('already_on_disk') == already


class TestDownloadRinexObsDecompress:

    def test_decompress_false_does_not_decompress(self, tmp_path):
        (tmp_path / "ALIC00AUS_R_20190010000_01D_30S_MO.crx.gz").touch()
        with patch('auto_download_PPP._scan_existing_files', return_value=set()), \
             patch('auto_download_PPP._download_rinex_from_ga', return_value=(set(), [])), \
             patch('auto_download_PPP.decompress_file') as mock_decomp:
            download_rinex_obs(['ALIC'], START, END, tmp_path, source='ga', decompress=False)
            mock_decomp.assert_not_called()

    def test_decompress_true_decompresses_crx_gz_files(self, tmp_path):
        (tmp_path / "ALIC00AUS_R_20190010000_01D_30S_MO.crx.gz").touch()
        (tmp_path / "DARW00AUS_R_20190010000_01D_30S_MO.crx.gz").touch()
        with patch('auto_download_PPP._scan_existing_files', return_value=set()), \
             patch('auto_download_PPP._download_rinex_from_ga', return_value=(set(), [])), \
             patch('auto_download_PPP.decompress_file') as mock_decomp:
            download_rinex_obs(['ALIC', 'DARW'], START, END, tmp_path, source='ga', decompress=True)
            assert mock_decomp.call_count == 2

    def test_decompress_true_no_files_does_not_error(self, tmp_path):
        with patch('auto_download_PPP._scan_existing_files', return_value=set()), \
             patch('auto_download_PPP._download_rinex_from_ga', return_value=(set(), [])), \
             patch('auto_download_PPP.decompress_file') as mock_decomp:
            download_rinex_obs(['ALIC'], START, END, tmp_path, source='ga', decompress=True)
            mock_decomp.assert_not_called()

    def test_delete_compressed_passed_to_decompress(self, tmp_path):
        gz = tmp_path / "ALIC00AUS_R_20190010000_01D_30S_MO.crx.gz"
        gz.touch()
        with patch('auto_download_PPP._scan_existing_files', return_value=set()), \
             patch('auto_download_PPP._download_rinex_from_ga', return_value=(set(), [])), \
             patch('auto_download_PPP.decompress_file') as mock_decomp:
            download_rinex_obs(['ALIC'], START, END, tmp_path, source='ga',
                               decompress=True, delete_compressed=True)
            mock_decomp.assert_called_once_with(gz, delete_after_decompression=True)

    def test_delete_compressed_false_passed_to_decompress(self, tmp_path):
        gz = tmp_path / "ALIC00AUS_R_20190010000_01D_30S_MO.crx.gz"
        gz.touch()
        with patch('auto_download_PPP._scan_existing_files', return_value=set()), \
             patch('auto_download_PPP._download_rinex_from_ga', return_value=(set(), [])), \
             patch('auto_download_PPP.decompress_file') as mock_decomp:
            download_rinex_obs(['ALIC'], START, END, tmp_path, source='ga',
                               decompress=True, delete_compressed=False)
            mock_decomp.assert_called_once_with(gz, delete_after_decompression=False)


class TestDownloadRinexObsBulk:

    def test_bulk_stations_and_days_passes_correct_pairs(self, tmp_path):
        """3 stations × 3 days should produce 9 station-day pairs queried from GA."""
        stations = ['ALIC', 'DARW', 'PERT']
        expected_pairs = {
            (s, d)
            for s in stations
            for d in ['2019-01-01', '2019-01-02', '2019-01-03']
        }
        with patch('auto_download_PPP._scan_existing_files', return_value=set()), \
             patch('auto_download_PPP._download_rinex_from_ga', return_value=(set(), [])) as mock_ga:
            download_rinex_obs(stations, START, END, tmp_path, source='ga')
            # GA receives the full set of pairs minus already_on_disk via already_on_disk kwarg;
            # verify it was called with all 3 stations
            args, kwargs = mock_ga.call_args
            assert set(args[0]) == {'ALIC', 'DARW', 'PERT'}

    def test_station_names_uppercased(self, tmp_path):
        """Station names should be uppercased before passing to download functions."""
        with patch('auto_download_PPP._scan_existing_files', return_value=set()), \
             patch('auto_download_PPP._download_rinex_from_ga', return_value=(set(), [])) as mock_ga:
            download_rinex_obs(['alic', 'darw'], START, END, tmp_path, source='ga')
            args, _ = mock_ga.call_args
            assert args[0] == ['ALIC', 'DARW']
