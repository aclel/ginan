"""Tests for auto_download_PPP.py"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

from auto_download_PPP import download_gnss_data_entry, download_files_from_gnss_data


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

                # Should only call download once when successful
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
                with patch('auto_download_PPP.sleep'):  # Speed up test
                    mock_download.side_effect = Exception("Network error")

                    result = download_gnss_data_entry(
                        entry=entry,
                        output_dir=Path(temp_dir),
                        max_retries=3,
                        if_file_present="replace"
                    )

                    # Should retry max_retries + 1 times (0, 1, 2, 3)
                    assert mock_download.call_count == 4
                    assert result is None


class TestDownloadFilesFromGnssData:

    def test_file_validation_passes_with_all_files(self):
        """Test validation passes when all station files exist"""
        station_list = ["ALIC", "DARW"]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock station files
            (temp_path / "ALIC00AUS_R_20191990000_01D_30S_MO.rnx").touch()
            (temp_path / "DARW00AUS_R_20191990000_01D_30S_MO.rnx").touch()

            with patch('auto_download_PPP.requests.get') as mock_get:
                mock_response = Mock()
                mock_response.content = '[]'
                mock_get.return_value = mock_response

                # Should complete without exception
                download_files_from_gnss_data(
                    station_list=station_list,
                    start_epoch=Mock(),
                    end_epoch=Mock(),
                    data_dir=temp_path
                )

    def test_file_validation_missing_files(self):
        """Test validation raises error when files are missing"""
        station_list = ["ALIC", "DARW", "MISS"]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files for only some stations
            (temp_path / "ALIC00AUS_R_20191990000_01D_30S_MO.rnx").touch()
            (temp_path / "DARW00AUS_R_20191990000_01D_30S_MO.rnx").touch()
            # MISS station has no files

            with patch('auto_download_PPP.requests.get') as mock_get:
                mock_response = Mock()
                mock_response.content = '[]'
                mock_get.return_value = mock_response

                with pytest.raises(FileNotFoundError, match="Required files missing"):
                    download_files_from_gnss_data(
                        station_list=station_list,
                        start_epoch=Mock(),
                        end_epoch=Mock(),
                        data_dir=temp_path
                    )
