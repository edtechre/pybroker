"""Tests for multi-interval compression."""

import numpy as np
import pandas as pd
import pytest

from .fixtures import *  # noqa: F401,F403
from pybroker.common import DataCol, IndicatorSymbol
from pybroker.indicator import IndicatorsMixin, indicator
from pybroker.model import model
from pybroker.scope import StaticScope
from pybroker.interval import (
    IntervalData,
    _normalize_duration_string,
    _validate_symbol_dates_for_base,
    build_compressed_symbol_df,
    compress,
    compress_bars,
    compress_symbol_df,
    compress_symbol_from_frame,
    compress_symbol_intervals_from_frame,
    compress_intervals_from_frame,
    compressed_bars_to_bar_data,
    format_interval,
    indicator_interval_name,
    is_valid_interval,
    lookahead_train_dates,
    model_interval_name,
    normalize_interval,
    parse_indicator_interval_name,
    parse_model_interval_name,
    slice_arrays_by_dates,
    slice_compressed_df_by_dates,
    validate_base_timeframe_data,
    validate_interval,
)


def _daily_bars(dates, o, h, low, c, v=None):
    n = len(dates)
    dates = np.array(dates, dtype="datetime64[D]")
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    v = (
        np.ones(n, dtype=np.float64)
        if v is None
        else np.asarray(v, dtype=np.float64)
    )
    return dates, o, h, low, c, v


def _minute_sym_df(periods: int = 390) -> pd.DataFrame:
    """Synthetic 1-minute OHLCV (~one trading day when periods=390)."""
    dates = pd.date_range("2020-01-06 09:30", periods=periods, freq="1min")
    close = np.linspace(100, 110, periods)
    return pd.DataFrame(
        {
            DataCol.DATE.value: dates,
            DataCol.OPEN.value: close,
            DataCol.HIGH.value: close + 0.5,
            DataCol.LOW.value: close - 0.5,
            DataCol.CLOSE.value: close,
            DataCol.VOLUME.value: np.ones(periods),
        }
    )


def _ohlcv_from_dates(dates: pd.DatetimeIndex) -> tuple:
    n = len(dates)
    close = np.arange(n, dtype=np.float64) + 1
    o = close
    h = close + 1
    low = close - 1
    v = np.ones(n)
    return (
        dates.to_numpy().astype("datetime64[ns]"),
        o,
        h,
        low,
        close,
        v,
    )


ONE_MINUTE_BASE_SECONDS = 60.0
DAILY_BASE_SECONDS = 86400.0

CALENDAR_INTERVALS = ("daily", "weekly", "monthly", "quarterly", "yearly")
DURATION_UNIT_LETTERS = ("s", "m", "h", "d")


class TestIntervals:
    @pytest.mark.parametrize("interval", [2, 5, 10])
    def test_normalize_every_n_bars(self, interval):
        assert normalize_interval(interval) == interval

    @pytest.mark.parametrize("interval", CALENDAR_INTERVALS)
    def test_normalize_calendar(self, interval):
        assert normalize_interval(interval) == interval

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("5m", "5m"),
            ("1h", "1h"),
            ("30s", "30s"),
            ("5d", "5d"),
            ("7d", "7d"),
            ("5M", "5m"),
            ("1H", "1h"),
        ],
    )
    def test_normalize_duration(self, raw, expected):
        assert normalize_interval(raw) == expected  # type: ignore[arg-type]

    @pytest.mark.parametrize("letter", DURATION_UNIT_LETTERS)
    def test_normalize_duration_unit_letters(self, letter):
        interval = f"1{letter}"
        assert normalize_interval(interval) == interval  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "value", [["weekly"], ("weekly",), 2.0, None, {"weekly"}]
    )
    def test_normalize_rejects_non_int_str(self, value):
        with pytest.raises(ValueError, match="Invalid interval"):
            normalize_interval(value)  # type: ignore[arg-type]

    def test_duration_normalization_memoized(self):
        _normalize_duration_string.cache_clear()
        assert normalize_interval("5m") == "5m"  # type: ignore[arg-type]
        assert normalize_interval("5m") == "5m"  # type: ignore[arg-type]
        assert _normalize_duration_string.cache_info().hits >= 1

    def test_invalid_duration_raises_repeatedly(self):
        # lru_cache does not cache exceptions; the error must repeat.
        for _ in range(2):
            with pytest.raises(ValueError, match="Invalid interval"):
                normalize_interval("5x")  # type: ignore[arg-type]

    def test_uppercase_duration_canonicalized_when_cached(self):
        _normalize_duration_string.cache_clear()
        for _ in range(2):
            assert normalize_interval("5M") == "5m"  # type: ignore[arg-type]

    def test_reject_one(self):
        with pytest.raises(ValueError, match="n > 1"):
            normalize_interval(1)

    def test_reject_multi_unit_duration(self):
        with pytest.raises(ValueError, match="Invalid interval"):
            normalize_interval("1h 30m")  # type: ignore[arg-type]

    def test_reject_long_unit_names(self):
        with pytest.raises(ValueError, match="Invalid interval"):
            normalize_interval("5min")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "raw,expected_days",
        [("1w", "7d"), ("2w", "14d")],
    )
    def test_reject_week_duration(self, raw, expected_days):
        with pytest.raises(
            ValueError,
            match=f"use 'weekly' for calendar weeks or '{expected_days}'",
        ):
            normalize_interval(raw)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Invalid interval"):
            normalize_interval("1hour")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "interval,expected",
        [
            (5, "5"),
            ("weekly", "weekly"),
            ("5m", "5m"),
            ("1h", "1h"),
            ("monthly", "monthly"),
        ],
    )
    def test_format_interval(self, interval, expected):
        assert format_interval(interval) == expected

    @pytest.mark.parametrize(
        "interval,suffix",
        [
            (5, "5"),
            ("weekly", "weekly"),
            ("5m", "5m"),
            ("1h", "1h"),
            ("monthly", "monthly"),
        ],
    )
    def test_indicator_and_parse_names(self, interval, suffix):
        assert indicator_interval_name("sma20", interval) == f"sma20@{suffix}"
        assert parse_indicator_interval_name(f"sma20@{suffix}") == (
            "sma20",
            interval,
        )
        assert parse_model_interval_name(f"m@{suffix}") == ("m", interval)


class TestIntervalHierarchyOnOneMinute:
    """Combinatoric granularity checks on a 1-minute base feed."""

    @pytest.mark.parametrize(
        "interval",
        ["weekly", 5, "1h"],
        ids=["weekly", "every_5_bars", "1h"],
    )
    def test_declared_combo_valid_on_one_minute(self, interval):
        validate_interval(interval, ONE_MINUTE_BASE_SECONDS)
        data = compress_symbol_df(
            _minute_sym_df(), interval, frozenset(), ONE_MINUTE_BASE_SECONDS
        )
        assert len(data.bars.close) > 0

    @pytest.mark.parametrize(
        "interval",
        [5, "2m", "5m", "1h", "1d", "daily", "weekly", "monthly"],
        ids=[
            "every_5_bars",
            "2m",
            "5m",
            "1h",
            "1d",
            "daily",
            "weekly",
            "monthly",
        ],
    )
    def test_coarser_intervals_accepted_on_one_minute(self, interval):
        validate_interval(interval, ONE_MINUTE_BASE_SECONDS)
        assert is_valid_interval(interval, ONE_MINUTE_BASE_SECONDS)

    @pytest.mark.parametrize(
        "interval",
        ["30s", "1m", "45s"],
        ids=["30s", "1m", "45s"],
    )
    def test_finer_intervals_rejected_on_one_minute(self, interval):
        with pytest.raises(ValueError, match="Cannot compress"):
            validate_interval(interval, ONE_MINUTE_BASE_SECONDS)

    def test_coarser_intervals_yield_fewer_or_equal_bars(self):
        sym_df = _minute_sym_df(periods=390)
        bar_counts = [
            len(
                compress_symbol_df(
                    sym_df, interval, frozenset(), ONE_MINUTE_BASE_SECONDS
                ).bars.close
            )
            for interval in (5, "5m", "1h", "weekly")
        ]
        every5, five_m, one_h, weekly = bar_counts
        assert weekly <= one_h <= five_m
        assert five_m <= every5

    def test_validate_base_from_one_minute_df(self):
        sym_df = _minute_sym_df(periods=10)
        validate_base_timeframe_data(sym_df, ONE_MINUTE_BASE_SECONDS)


class TestCompressAllCalendarIntervals:
    @pytest.mark.parametrize(
        "interval,dates",
        [
            ("daily", pd.date_range("2020-01-01", periods=5, freq="D")),
            ("weekly", pd.date_range("2020-01-06", periods=10, freq="B")),
            ("monthly", pd.date_range("2020-01-01", periods=90, freq="D")),
            ("quarterly", pd.date_range("2020-01-01", periods=400, freq="D")),
            ("yearly", pd.date_range("2018-01-01", periods=800, freq="D")),
        ],
        ids=["daily", "weekly", "monthly", "quarterly", "yearly"],
    )
    def test_compress_calendar_produces_bars(self, interval, dates):
        dates_arr, o, h, low, c, v = _ohlcv_from_dates(dates)
        bars, completed = compress(dates_arr, o, h, low, c, v, interval)
        assert len(bars.close) >= 1
        assert len(completed) == len(dates)
        assert bars.open[0] == o[0]
        assert bars.close[-1] == c[-1]


class TestCompressAllDurationIntervals:
    @pytest.mark.parametrize(
        "interval,periods,expected_bins",
        [
            ("30s", 4, 4),
            ("1m", 4, 4),
            ("5m", 10, 2),
            ("1h", 120, 3),
        ],
        ids=["30s", "1m", "5m", "1h"],
    )
    def test_compress_duration_on_one_minute_base(
        self, interval, periods, expected_bins
    ):
        dates = pd.date_range("2020-01-01 09:30", periods=periods, freq="1min")
        dates_arr, o, h, low, c, v = _ohlcv_from_dates(dates)
        bars, _ = compress(dates_arr, o, h, low, c, v, interval)
        assert len(bars.close) == expected_bins

    def test_compress_one_day_duration_on_daily_base(self):
        dates, o, h, low, c, v = _daily_bars(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [0.5, 1.5, 2.5, 3.5],
            [1.5, 2.5, 3.5, 4.5],
        )
        bars, _ = compress(dates, o, h, low, c, v, "1d")
        assert len(bars.close) == 4

    def test_compress_seven_day_duration_on_daily_base(self):
        dates = pd.date_range("2020-01-06", periods=14, freq="D")
        dates_arr, o, h, low, c, v = _ohlcv_from_dates(dates)
        bars, _ = compress(dates_arr, o, h, low, c, v, "7d")
        assert len(bars.close) == 3


class TestValidateIntervalGranularity:
    @pytest.mark.parametrize(
        "base_bar_seconds,interval",
        [
            (60, "5m"),
            (60, "1h"),
            (60, 5),
            (300, "1h"),
            (86400, "weekly"),
            (86400, 5),
            (86400, "monthly"),
        ],
        ids=[
            "1m_to_5m",
            "1m_to_1h",
            "1m_to_5bars",
            "5m_to_1h",
            "daily_to_weekly",
            "daily_to_5bars",
            "daily_to_monthly",
        ],
    )
    def test_validate_interval_accepts_valid(self, base_bar_seconds, interval):
        validate_interval(interval, base_bar_seconds)
        assert is_valid_interval(interval, base_bar_seconds)

    @pytest.mark.parametrize(
        "base_bar_seconds,interval",
        [
            (86400, "daily"),
            (86400, "1m"),
            (86400, "1h"),
            (604800, "daily"),
            (604800, "weekly"),
            (28 * 86400, "weekly"),
            (300, "1m"),
        ],
        ids=[
            "daily_to_daily",
            "daily_to_1m",
            "daily_to_1h",
            "weekly_to_daily",
            "weekly_to_weekly",
            "monthly_to_weekly",
            "5m_to_1m",
        ],
    )
    def test_validate_interval_rejects_invalid(
        self, base_bar_seconds, interval
    ):
        with pytest.raises(ValueError, match="Cannot compress"):
            validate_interval(interval, base_bar_seconds)

    def test_validate_weekly_on_daily(self):
        validate_interval("weekly", 86400)

    def test_reject_daily_on_daily(self):
        with pytest.raises(ValueError, match="Cannot compress daily bars"):
            validate_interval("daily", 86400)

    def test_reject_weekly_on_weekly(self):
        with pytest.raises(ValueError, match="Cannot compress weekly bars"):
            validate_interval("weekly", 604800)

    def test_reject_weekly_on_monthly(self):
        with pytest.raises(ValueError, match="Cannot compress monthly bars"):
            validate_interval("weekly", 28 * 86400)

    def test_reject_daily_on_weekly(self):
        with pytest.raises(ValueError, match="Cannot compress weekly bars"):
            validate_interval("daily", 604800)


class TestCompressEveryN:
    def test_ohlcv_aggregation(self):
        dates, o, h, low, c, v = _daily_bars(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [0.5, 1.5, 2.5, 3.5],
            [1.5, 2.5, 3.5, 4.5],
            [10, 20, 30, 40],
        )
        bars, completed = compress(dates, o, h, low, c, v, 2)
        assert len(bars.close) == 2
        assert bars.open[0] == 1
        assert bars.high[0] == 3
        assert bars.low[0] == 0.5
        assert bars.close[0] == 2.5
        assert bars.volume[0] == 30
        assert bars.open[1] == 3
        assert bars.close[1] == 4.5
        assert bars.volume[1] == 70
        assert list(completed) == [-1, 0, 0, 1]

    def test_single_bar_bins(self):
        dates, o, h, low, c, v = _daily_bars(
            ["2020-01-01", "2020-01-02"],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
        )
        bars, completed = compress(dates, o, h, low, c, v, 2)
        assert len(bars.close) == 1
        assert bars.open[0] == 1
        assert bars.close[0] == 2
        assert completed[0] == -1
        assert completed[1] == 0

    def test_final_partial_bin_invisible(self):
        dates, o, h, low, c, v = _daily_bars(
            ["2020-01-01", "2020-01-02", "2020-01-03"],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        )
        bars, completed = compress(dates, o, h, low, c, v, 2)
        assert len(bars.close) == 2
        assert completed[-1] == 0
        assert completed[-1] != 1

    def test_float_dtype(self):
        dates, o, h, low, c, v = _daily_bars(
            ["2020-01-01"], [1.5], [2.5], [0.5], [1.0]
        )
        bars, _ = compress(dates, o, h, low, c, v, 2)
        assert bars.close.dtype == np.float64

    def test_nan_volume(self):
        dates, o, h, low, c, _ = _daily_bars(
            ["2020-01-01", "2020-01-02"],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
        )
        v = np.array([np.nan, 1.0])
        bars, _ = compress(dates, o, h, low, c, v, 2)
        assert np.isnan(bars.volume[0])


class TestCompressWeekly:
    def test_iso_week_alignment(self):
        # Mon 2020-01-06 through Fri 2020-01-10 -> one week. The trailing Mon
        # starts a second bin, which closes the first one.
        dates = np.array(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-13",
            ],
            dtype="datetime64[D]",
        )
        o = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        h = o + 1
        low = o - 0.5
        c = o + 0.5
        v = np.ones(6)
        bars, completed = compress(dates, o, h, low, c, v, "weekly")
        assert len(bars.close) == 2
        assert bars.open[0] == 1
        assert bars.close[0] == 5.5
        assert bars.high[0] == 6
        assert bars.low[0] == 0.5
        assert bars.volume[0] == 5
        # The trailing bin is still forming, so it is never completed.
        assert list(completed) == [-1, -1, -1, -1, 0, 0]

    def test_holiday_split_week(self):
        # Mon-Thu only (4 bars) still one weekly bin
        dates = np.array(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-13",
            ],
            dtype="datetime64[D]",
        )
        c = np.array([10, 11, 12, 13, 14], dtype=np.float64)
        o = c - 1
        h = c + 1
        low = c - 2
        v = np.ones(5)
        bars, completed = compress(dates, o, h, low, c, v, "weekly")
        assert len(bars.close) == 2
        assert bars.close[0] == 13
        assert completed[3] == 0

    def test_mid_week_start(self):
        # Wed-Fri, then Mon-Tue, then the Mon that closes the second bin.
        dates = np.array(
            [
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-13",
                "2020-01-14",
                "2020-01-20",
            ],
            dtype="datetime64[D]",
        )
        c = np.arange(6, dtype=np.float64)
        o = c
        h = c + 1
        low = c - 1
        v = np.ones(6)
        bars, completed = compress(dates, o, h, low, c, v, "weekly")
        assert len(bars.close) == 3
        assert bars.close[0] == 2
        assert bars.close[1] == 4
        assert completed[2] == 0
        assert completed[4] == 1

    def test_completed_at_bin_boundary(self):
        dates = np.array(
            ["2020-01-06", "2020-01-07", "2020-01-13", "2020-01-20"],
            dtype="datetime64[D]",
        )
        c = np.array([1, 2, 3, 4], dtype=np.float64)
        o = h = low = c
        v = np.ones(4)
        bars, completed = compress(dates, o, h, low, c, v, "weekly")
        assert completed[1] == 0
        assert bars.close[completed[1]] == 2
        assert completed[2] == 1
        assert bars.close[completed[2]] == 3

    def test_final_bin_never_completed(self):
        # A calendar bin is only closed once a bar in a later bin arrives, so
        # the trailing bin -- complete or partial -- stays invisible.
        for periods in (8, 10):
            dates = (
                pd.date_range("2023-01-02", periods=periods, freq="B")
                .to_numpy()
                .astype("datetime64[ns]")
            )
            n = len(dates)
            c = np.arange(1.0, n + 1, dtype=np.float64)
            bars, completed = compress(
                dates,
                c,
                (c + 1).astype(np.float64),
                (c - 1).astype(np.float64),
                c,
                np.ones(n, dtype=np.float64),
                "weekly",
            )
            assert completed[-1] == len(bars.close) - 2

    def test_pandas_resample_reference(self):
        dates = pd.date_range("2020-01-06", periods=10, freq="B")
        n = len(dates)
        df = pd.DataFrame(
            {
                "open": np.arange(n, dtype=float),
                "high": np.arange(n, dtype=float) + 1,
                "low": np.arange(n, dtype=float) - 1,
                "close": np.arange(n, dtype=float) + 0.5,
                "volume": np.ones(n),
            },
            index=dates,
        )
        ref = (
            df.resample("W-FRI")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        bars, _ = compress(
            df.index.to_numpy().astype("datetime64[ns]"),
            df["open"].to_numpy(),
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["close"].to_numpy(),
            df["volume"].to_numpy(),
            "weekly",
        )
        assert len(bars.close) == len(ref)
        np.testing.assert_array_almost_equal(bars.open, ref["open"].to_numpy())
        np.testing.assert_array_almost_equal(bars.high, ref["high"].to_numpy())
        np.testing.assert_array_almost_equal(bars.low, ref["low"].to_numpy())
        np.testing.assert_array_almost_equal(
            bars.close, ref["close"].to_numpy()
        )
        np.testing.assert_array_almost_equal(
            bars.volume, ref["volume"].to_numpy()
        )


class TestValidateBaseTimeframeData:
    def test_accepts_matching_daily_spacing(self):
        df = pd.DataFrame(
            {
                DataCol.DATE.value: pd.date_range(
                    "2020-01-01", periods=5, freq="D"
                ),
                DataCol.SYMBOL.value: ["SPY"] * 5,
            }
        )
        validate_base_timeframe_data(df, DAILY_BASE_SECONDS)

    def test_rejects_mismatch(self):
        df = _minute_sym_df(periods=10)
        with pytest.raises(ValueError, match="inconsistent with base"):
            validate_base_timeframe_data(df, DAILY_BASE_SECONDS)


class TestCompressBars:
    def test_requires_base_timeframe(self):
        sym_df = _minute_sym_df(periods=10)
        with pytest.raises(TypeError):
            compress_bars(sym_df, "5m")  # type: ignore[call-arg]

    def test_returns_bar_data(self):
        sym_df = _minute_sym_df(periods=10)
        bars = compress_bars(sym_df, "5m", base_timeframe="1m")
        assert len(bars.close) == 2

    def test_indicator_callable(self):
        def sma(bar_data, period):
            return bar_data.close

        sym_df = _minute_sym_df(periods=10)
        sma_ind = indicator("sma", sma, period=5)
        weekly = compress_bars(sym_df, 5, base_timeframe="1m")
        assert len(sma_ind(weekly)) == 2


class TestCompressDuration:
    def test_five_minute_bins(self):
        dates = pd.date_range("2020-01-01 09:30", periods=11, freq="1min")
        n = len(dates)
        close = np.arange(n, dtype=np.float64) + 1
        o = close
        h = close + 1
        low = close - 1
        v = np.ones(n)
        bars, completed = compress(
            dates.to_numpy().astype("datetime64[ns]"),
            o,
            h,
            low,
            close,
            v,
            "5m",
        )
        assert len(bars.close) == 3
        assert bars.open[0] == 1
        assert bars.close[0] == 5
        assert bars.open[1] == 6
        assert bars.close[1] == 10
        # Bin 1 closes once the 11th bar opens bin 2; bin 2 is still forming.
        assert completed[-1] == 1


class TestCompressSymbolDfValidation:
    def _daily_sym_df(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                DataCol.DATE.value: dates,
                DataCol.OPEN.value: np.arange(5, dtype=float),
                DataCol.HIGH.value: np.arange(5, dtype=float) + 1,
                DataCol.LOW.value: np.arange(5, dtype=float) - 1,
                DataCol.CLOSE.value: np.arange(5, dtype=float) + 0.5,
                DataCol.VOLUME.value: np.ones(5),
            }
        )

    def test_rejects_invalid_calendar_interval(self):
        with pytest.raises(ValueError, match="Cannot compress daily bars"):
            compress_symbol_df(
                self._daily_sym_df(), "daily", frozenset(), DAILY_BASE_SECONDS
            )

    def test_allows_valid_calendar_interval(self):
        data = compress_symbol_df(
            self._daily_sym_df(), "weekly", frozenset(), DAILY_BASE_SECONDS
        )
        assert len(data.bars.close) > 0

    def test_compress_symbol_df_valid_duration(self):
        dates = pd.date_range("2020-01-01 09:30", periods=10, freq="1min")
        sym_df = pd.DataFrame(
            {
                DataCol.DATE.value: dates,
                DataCol.OPEN.value: np.arange(10, dtype=float),
                DataCol.HIGH.value: np.arange(10, dtype=float) + 1,
                DataCol.LOW.value: np.arange(10, dtype=float) - 1,
                DataCol.CLOSE.value: np.arange(10, dtype=float) + 0.5,
                DataCol.VOLUME.value: np.ones(10),
            }
        )
        data = compress_symbol_df(
            sym_df, "5m", frozenset(), ONE_MINUTE_BASE_SECONDS
        )
        assert len(data.bars.close) == 2


class TestBatchCompressionFromFrame:
    def _multi_symbol_df(self) -> pd.DataFrame:
        dates = pd.date_range("2020-01-06", periods=10, freq="B")
        frames = []
        for sym in ("SPY", "QQQ"):
            close = np.arange(10, dtype=np.float64) + (
                1 if sym == "SPY" else 11
            )
            frames.append(
                pd.DataFrame(
                    {
                        DataCol.SYMBOL.value: [sym] * 10,
                        DataCol.DATE.value: dates,
                        DataCol.OPEN.value: close,
                        DataCol.HIGH.value: close + 1,
                        DataCol.LOW.value: close - 1,
                        DataCol.CLOSE.value: close,
                        DataCol.VOLUME.value: np.ones(10),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def test_symbol_intervals_matches_single_interval_calls(self):
        df = self._multi_symbol_df()
        intervals = ("weekly", 2)
        custom_cols: frozenset[str] = frozenset()
        batch = compress_symbol_intervals_from_frame(
            df, "SPY", intervals, custom_cols, DAILY_BASE_SECONDS
        )
        for interval in intervals:
            single = compress_symbol_from_frame(
                df, "SPY", interval, custom_cols, DAILY_BASE_SECONDS
            )
            expected = batch[interval]
            np.testing.assert_array_equal(
                expected.bars.close, single.bars.close
            )
            np.testing.assert_array_equal(expected.completed, single.completed)
            np.testing.assert_array_equal(
                expected.base_dates, single.base_dates
            )

    def test_intervals_from_frame_matches_per_symbol_calls(self):
        df = self._multi_symbol_df()
        symbols = {"SPY", "QQQ"}
        intervals = frozenset({"weekly", 2})
        custom_cols: frozenset[str] = frozenset()
        batch = compress_intervals_from_frame(
            df,
            {sym: intervals for sym in symbols},
            custom_cols,
            DAILY_BASE_SECONDS,
        )
        expected = IntervalData()
        for sym in symbols:
            for interval in intervals:
                expected.compressed[(sym, interval)] = (
                    compress_symbol_from_frame(
                        df, sym, interval, custom_cols, DAILY_BASE_SECONDS
                    )
                )
        assert set(batch.compressed) == set(expected.compressed)
        for key, data in batch.compressed.items():
            ref = expected.compressed[key]
            np.testing.assert_array_equal(data.bars.close, ref.bars.close)
            np.testing.assert_array_equal(data.completed, ref.completed)

    def test_intervals_from_frame_narrows_per_symbol(self):
        df = self._multi_symbol_df()
        batch = compress_intervals_from_frame(
            df,
            {"SPY": frozenset({"weekly"}), "QQQ": frozenset({2})},
            frozenset(),
            DAILY_BASE_SECONDS,
        )
        assert set(batch.compressed) == {("SPY", "weekly"), ("QQQ", 2)}

    def test_intervals_from_frame_skips_undeclared_symbols(self):
        df = self._multi_symbol_df()
        batch = compress_intervals_from_frame(
            df, {"SPY": frozenset({"weekly"})}, frozenset(), DAILY_BASE_SECONDS
        )
        assert set(batch.compressed) == {("SPY", "weekly")}

    def test_intervals_from_frame_empty_mapping(self):
        df = self._multi_symbol_df()
        batch = compress_intervals_from_frame(
            df, {}, frozenset(), DAILY_BASE_SECONDS
        )
        assert not batch.compressed

    def test_intervals_from_frame_validates_bar_spacing(self):
        # Daily bars cannot come from a two-day base feed: the gaps are finer
        # than the declared spacing.
        df = self._multi_symbol_df()
        with pytest.raises(ValueError, match="inconsistent with base"):
            compress_intervals_from_frame(
                df,
                {"SPY": frozenset({"weekly"})},
                frozenset(),
                2 * DAILY_BASE_SECONDS,
            )

    def test_intervals_from_frame_skips_spacing_of_undeclared_symbols(self):
        # QQQ is not compressed, so its spacing is never validated.
        df = self._multi_symbol_df()
        df = df[
            (df[DataCol.SYMBOL.value] == "SPY")
            | (df[DataCol.DATE.value] != df[DataCol.DATE.value].iloc[1])
        ].reset_index(drop=True)
        batch = compress_intervals_from_frame(
            df, {"SPY": frozenset({"weekly"})}, frozenset(), DAILY_BASE_SECONDS
        )
        assert set(batch.compressed) == {("SPY", "weekly")}


class TestModelIntervalNaming:
    def test_model_interval_name(self):
        assert model_interval_name("m", "weekly") == "m@weekly"

    def test_parse_model_interval_name(self):
        assert parse_model_interval_name("m@weekly") == ("m", "weekly")
        assert parse_model_interval_name("m") == ("m", None)


class TestBuildCompressedSymbolDf:
    def test_build_and_slice_by_dates(self):
        sym = "SPY"
        dates = np.array(
            ["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"],
            dtype="datetime64[D]",
        )
        n = len(dates)
        close = np.arange(n, dtype=np.float64) + 1
        df = pd.DataFrame(
            {
                DataCol.SYMBOL.value: [sym] * n,
                DataCol.DATE.value: dates,
                DataCol.OPEN.value: close,
                DataCol.HIGH.value: close + 1,
                DataCol.LOW.value: close - 1,
                DataCol.CLOSE.value: close,
                DataCol.VOLUME.value: np.ones(n),
            }
        )
        compressed = compress_symbol_df(df, 2, frozenset(), DAILY_BASE_SECONDS)
        ind_series = pd.Series(
            np.array([10.0, 20.0]), index=compressed.bars.dates
        )
        built = build_compressed_symbol_df(
            sym,
            2,
            compressed,
            {IndicatorSymbol("sma@2", sym): ind_series},
            ("sma",),
            frozenset(),
        )
        assert list(built.columns) == [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "sma",
        ]
        sliced = slice_compressed_df_by_dates(built, dates[:2])
        assert len(sliced) == 1


LOOKAHEAD_INTERVALS = ("weekly", "monthly", 5)


class TestLookaheadTrainDates:
    def _compressed(self, interval, periods=420):
        """Compresses ``periods`` business days; returns base dates + bars."""
        dates = pd.date_range("2020-01-06", periods=periods, freq="B")
        d, o, h, low, c, v = _ohlcv_from_dates(dates)
        bars, _ = compress(d, o, h, low, c, v, interval)
        return d, bars

    def _slice_close(self, bars, selected):
        columns = (DataCol.CLOSE.value,)
        arrays = {DataCol.CLOSE.value: bars.close}
        _, sliced, sliced_dates = slice_arrays_by_dates(
            columns, arrays, bars.dates, selected
        )
        return sliced[DataCol.CLOSE.value], sliced_dates

    @pytest.mark.parametrize("interval", LOOKAHEAD_INTERVALS)
    def test_lookahead_one_is_passthrough(self, interval):
        d, bars = self._compressed(interval)
        train_dates, test_dates = d[:300], d[300:]
        result, dropped = lookahead_train_dates(
            bars.dates, train_dates, test_dates, 1
        )
        assert dropped == 0
        expected_close, expected_dates = self._slice_close(bars, train_dates)
        actual_close, actual_dates = self._slice_close(bars, result)
        assert len(expected_dates) > 0
        np.testing.assert_array_equal(actual_dates, expected_dates)
        np.testing.assert_array_equal(actual_close, expected_close)

    @pytest.mark.parametrize("lookahead", (1, 2, 3, 5))
    @pytest.mark.parametrize("interval", LOOKAHEAD_INTERVALS)
    def test_compressed_gap_equals_lookahead(self, interval, lookahead):
        d, bars = self._compressed(interval)
        train_dates, test_dates = d[:300], d[300:]
        result, dropped = lookahead_train_dates(
            bars.dates, train_dates, test_dates, lookahead
        )
        kept_idx = np.nonzero(np.isin(bars.dates, result))[0]
        test_idx = np.nonzero(np.isin(bars.dates, test_dates))[0]
        train_idx = np.nonzero(np.isin(bars.dates, train_dates))[0]
        assert kept_idx.size > 0
        assert test_idx[0] - kept_idx[-1] == lookahead
        if lookahead <= 1:
            assert dropped == 0
        else:
            assert dropped == train_idx.size - kept_idx.size

    def test_empty_test_dates_returns_selection_unchanged(self):
        d, bars = self._compressed("weekly")
        train_dates = d[:300]
        empty = np.array([], dtype="datetime64[ns]")
        result, dropped = lookahead_train_dates(
            bars.dates, train_dates, empty, 3
        )
        assert dropped == 0
        train_idx = np.nonzero(np.isin(bars.dates, train_dates))[0]
        assert train_idx.size > 0
        np.testing.assert_array_equal(result, bars.dates[train_idx])
        expected_close, expected_dates = self._slice_close(bars, train_dates)
        actual_close, actual_dates = self._slice_close(bars, result)
        np.testing.assert_array_equal(actual_dates, expected_dates)
        np.testing.assert_array_equal(actual_close, expected_close)

    def test_empty_train_dates_returns_empty(self):
        d, bars = self._compressed("weekly")
        empty = np.array([], dtype="datetime64[ns]")
        result, dropped = lookahead_train_dates(bars.dates, empty, d[300:], 3)
        assert dropped == 0
        assert result.size == 0

    def test_lookahead_exceeds_train_span_drops_all(self):
        d, bars = self._compressed("weekly", periods=60)
        train_dates, test_dates = d[:30], d[30:]
        train_idx = np.nonzero(np.isin(bars.dates, train_dates))[0]
        assert train_idx.size > 0
        result, dropped = lookahead_train_dates(
            bars.dates, train_dates, test_dates, 50
        )
        assert result.size == 0
        assert dropped == train_idx.size

    def test_empty_bar_dates_passthrough(self):
        d = (
            pd.date_range("2020-01-06", periods=10, freq="B")
            .to_numpy()
            .astype("datetime64[ns]")
        )
        empty = np.array([], dtype="datetime64[ns]")
        result, dropped = lookahead_train_dates(empty, d[:6], d[6:], 5)
        assert dropped == 0
        np.testing.assert_array_equal(result, d[:6])

    @pytest.mark.parametrize("interval", LOOKAHEAD_INTERVALS)
    def test_kept_indices_are_contiguous_prefix(self, interval):
        d, bars = self._compressed(interval)
        train_dates, test_dates = d[:300], d[300:]
        result, dropped = lookahead_train_dates(
            bars.dates, train_dates, test_dates, 3
        )
        train_idx = np.nonzero(np.isin(bars.dates, train_dates))[0]
        kept_idx = np.nonzero(np.isin(bars.dates, result))[0]
        positions = np.nonzero(np.isin(train_idx, kept_idx))[0]
        assert positions.size > 0
        # Only a suffix of the compressed train history is dropped: the
        # kept bars are a contiguous prefix, so lag features keep their
        # per-bar offsets intact.
        assert positions[0] == 0
        assert np.all(np.diff(positions) == 1)
        assert positions[-1] == train_idx.size - dropped - 1


class TestIntervalIndicatorCompute:
    def test_ind_interval_matches_hand_compressed(self):
        def sma(bar_data, period):
            close = bar_data.close
            out = np.full(len(close), np.nan)
            for i in range(period - 1, len(close)):
                out[i] = np.mean(close[i - period + 1 : i + 1])
            return out

        StaticScope.set_instance(None)
        sma_ind = indicator("sma20", sma, period=5)
        sym = "SPY"
        dates = np.array(
            [
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ],
            dtype="datetime64[D]",
        )
        n = len(dates)
        close = np.arange(n, dtype=np.float64) + 1
        df = pd.DataFrame(
            {
                DataCol.SYMBOL.value: [sym] * n,
                DataCol.DATE.value: dates,
                DataCol.OPEN.value: close,
                DataCol.HIGH.value: close + 1,
                DataCol.LOW.value: close - 1,
                DataCol.CLOSE.value: close,
                DataCol.VOLUME.value: np.ones(n),
            }
        )
        interval_data = IntervalData()
        sym_df = df.reset_index(drop=True)
        weekly_name = indicator_interval_name("sma20", "weekly")
        interval_data.compressed[(sym, "weekly")] = compress_symbol_df(
            sym_df, "weekly", frozenset(), DAILY_BASE_SECONDS
        )
        mixin = IndicatorsMixin()
        result = mixin.compute_indicators(
            df=df,
            indicator_syms=[IndicatorSymbol(weekly_name, sym)],
            cache_date_fields=None,
            parallel_indicators=False,
            interval_data=interval_data,
        )
        hand = sma_ind(
            compressed_bars_to_bar_data(
                interval_data.compressed[(sym, "weekly")].bars
            )
        )
        key = IndicatorSymbol(weekly_name, sym)
        np.testing.assert_array_almost_equal(
            result[key].to_numpy(), hand.to_numpy()
        )


class TestReservedNameSeparator:
    @pytest.mark.parametrize(
        "name", ["vol@30", "sma@2", "close@1h", "spread@weekly", "a@b"]
    )
    def test_indicator_rejects_at_sign(self, name, scope):
        with pytest.raises(ValueError, match="reserved for interval"):
            indicator(name, lambda bar_data: bar_data.close)

    def test_model_rejects_at_sign(self, scope):
        with pytest.raises(ValueError, match="reserved for interval"):
            model("m@5", lambda sym, train_data, test_data: None)

    def test_plain_names_still_accepted(self, scope):
        assert indicator("sma20", lambda bar_data: bar_data.close).name == (
            "sma20"
        )


class TestBaseSpacingValidation:
    def _dates(self, values):
        return np.array(values, dtype="datetime64[ns]")

    @pytest.mark.parametrize(
        "label,values,base",
        [
            (
                "weekend gap",
                ["2020-01-02", "2020-01-03", "2020-01-06"],
                86400.0,
            ),
            (
                "dst shift",
                ["2023-03-10T05:00", "2023-03-13T04:00", "2023-03-14T04:00"],
                86400.0,
            ),
            (
                "days filter",
                ["2020-01-06", "2020-01-13", "2020-01-20"],
                86400.0,
            ),
            (
                "sparse minutes",
                [
                    "2020-01-01T09:30",
                    "2020-01-01T09:32",
                    "2020-01-01T09:34",
                ],
                60.0,
            ),
            ("single bar", ["2020-01-02"], 86400.0),
        ],
    )
    def test_accepts_multiples_of_base(self, label, values, base):
        _validate_symbol_dates_for_base(label, self._dates(values), base)

    @pytest.mark.parametrize(
        "values,base",
        [
            (
                [
                    "2020-01-01T09:30:00",
                    "2020-01-01T09:30:30",
                    "2020-01-01T09:31:00",
                ],
                60.0,
            ),
            (
                [
                    "2020-01-01T09:30:00",
                    "2020-01-01T09:31:30",
                    "2020-01-01T09:33:00",
                ],
                60.0,
            ),
        ],
    )
    def test_rejects_non_multiples(self, values, base):
        with pytest.raises(ValueError, match="inconsistent with base"):
            _validate_symbol_dates_for_base("X", self._dates(values), base)


class TestCompressBarsSymbolGuard:
    def _sym_df(self, symbol):
        dates = pd.date_range("2020-01-06", periods=10, freq="B")
        close = np.arange(1.0, len(dates) + 1)
        return pd.DataFrame(
            {
                "symbol": symbol,
                "date": dates,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(len(dates)),
            }
        )

    def test_multi_symbol_frame_raises(self):
        df = pd.concat([self._sym_df("AAA"), self._sym_df("BBB")])
        with pytest.raises(ValueError, match="single symbol"):
            compress_bars(df, "weekly", base_timeframe="1d")

    def test_single_symbol_frame_allowed(self):
        bars = compress_bars(
            self._sym_df("AAA"), "weekly", base_timeframe="1d"
        )
        assert len(bars.close) > 0


class TestCompressVwap:
    def _df(self, volume):
        dates = pd.date_range("2020-01-06", periods=10, freq="B")
        close = np.arange(1.0, len(dates) + 1)
        return pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": volume,
                "vwap": close + 0.5,
            }
        )

    def test_volume_weighted(self):
        volume = np.arange(1.0, 11.0)
        bars = compress_bars(self._df(volume), "weekly", base_timeframe="1d")
        week = np.arange(1.0, 6.0)
        expected = ((week + 0.5) * week).sum() / week.sum()
        assert bars.vwap is not None
        assert bars.vwap[0] == pytest.approx(expected)

    def test_zero_volume_falls_back_to_mean(self):
        bars = compress_bars(
            self._df(np.zeros(10)), "weekly", base_timeframe="1d"
        )
        assert bars.vwap is not None
        assert bars.vwap[0] == pytest.approx(
            (np.arange(1.0, 6.0) + 0.5).mean()
        )

    def test_missing_vwap_column_stays_none(self):
        df = self._df(np.ones(10)).drop(columns=["vwap"])
        assert compress_bars(df, "weekly", base_timeframe="1d").vwap is None


@pytest.mark.parametrize("nan_at", [0, 1, 2])
def test_compress_bars_skips_nan_high_low_wherever_it_falls(nan_at):
    """A NaN high/low must be skipped consistently within a bin.

    Seeding the running extremes from the bin's first bar makes a NaN there
    swallow the whole bin, because every later ``>``/``<`` against NaN is
    False, while a NaN anywhere else is skipped by that same comparison.
    """
    dates = pd.date_range("2021-01-04", periods=6, freq="D")
    rows = []
    for i, date in enumerate(dates):
        rows.append(
            dict(
                symbol="A",
                date=date,
                open=5.0,
                high=np.nan if i == nan_at else 10.0 + i,
                low=np.nan if i == nan_at else 1.0 + i,
                close=5.0,
                volume=100.0,
            )
        )
    bars = compress_bars(pd.DataFrame(rows), "3d", base_timeframe="1d")
    assert not np.isnan(bars.high).any()
    assert not np.isnan(bars.low).any()


def test_compress_every_n_bars_skips_nan_like_calendar_bins():
    """Int intervals must aggregate high/low the same way calendar bins do.

    _aggregate_bins deliberately skips NaN so the result does not depend on
    where in the bin the gap falls; the reshape fast path propagated it, so
    one bad tick poisoned the whole compressed bar -- and NaN comparisons fail
    closed, silently taking the wrong branch.
    """
    # 11 bars at interval 3: three full bins plus a ragged tail of two.
    n = 11
    dates = (
        np.arange(n).astype("timedelta64[D]") + np.datetime64("2021-01-04")
    ).astype("datetime64[ns]")
    high = np.arange(1.0, n + 1)
    low = -np.arange(1.0, n + 1)
    high[1] = np.nan
    low[1] = np.nan
    # Also gap one bar of the ragged tail bin, which still has a finite one.
    high[9] = np.nan
    low[9] = np.nan
    close = np.arange(1.0, n + 1)
    bars, _ = compress(
        dates,
        np.arange(1.0, n + 1),
        high,
        low,
        close,
        np.ones(n),
        interval=3,
    )
    assert not np.isnan(bars.high[0])
    assert bars.high[0] == 3.0
    assert bars.low[0] == -3.0
    # The ragged tail bin holds one NaN beside one finite bar.
    assert bars.high[-1] == 11.0
    assert bars.low[-1] == -11.0
