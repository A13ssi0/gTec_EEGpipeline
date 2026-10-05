"""Low-overhead, optional timing telemetry for the real-time pipeline.

The monitor never changes EEG payloads or socket messages.  It reports cadence,
processing time, and the age of the timestamp already present in TCP messages.
Timestamp age is an absolute inter-machine latency only when machine clocks are
synchronised; cadence and processing measurements are local and monotonic.
"""

from datetime import datetime, timedelta
import threading
import time


def timestamp_age_ms(timestamp):
    """Return the wall-clock age of a pipeline timestamp, handling midnight."""
    try:
        sent_time = datetime.strptime(timestamp, "%H:%M:%S.%f").time()
        now = datetime.now()
        sent = now.replace(hour=sent_time.hour, minute=sent_time.minute,
                           second=sent_time.second, microsecond=sent_time.microsecond)
        if sent - now > timedelta(hours=12):
            sent -= timedelta(days=1)
        elif now - sent > timedelta(hours=12):
            sent += timedelta(days=1)
        return (now - sent).total_seconds() * 1000
    except (TypeError, ValueError):
        return None


class PipelineTelemetry:
    """Aggregate stream timing and report compact summaries at a bounded rate."""

    def __init__(self, name, expected_period_s=None, enabled=True,
                 report_interval_s=5, verbose=False,
                 rate_tolerance=0.02, max_gap_ms=250, max_input_age_ms=100, check_cadence=True):
        self.name = name
        # Only the source (Acquisition) checks rate and pauses: downstream nodes inherit them, so they would repeat
        # the same warning. A downstream node falling behind shows up as its inputs waiting (max_input_age_ms)
        self.check_cadence = check_cadence
        # Warning tolerances: jitter is normal (Bluetooth delivers chunks in bursts), falling behind is not
        self.rate_tolerance = rate_tolerance        # average rate may be this fraction below the expected one
        self.max_gap_ms = max_gap_ms                # longest acceptable pause between two chunks
        self.max_input_age_ms = max_input_age_ms    # p95 of how long a message waited before being processed
        self.expected_period_s = expected_period_s
        self.enabled = enabled
        self.verbose = verbose
        self.report_interval_s = float(report_interval_s)
        self.report_interval_s = max(1.0, self.report_interval_s)
        self._lock = threading.Lock()
        self._last_tick = None
        self._last_report = time.perf_counter()
        self._intervals_s = []
        self._processing_ms = []
        self._transport_ms = []
        self._events = 0

    def set_expected_period(self, expected_period_s):
        self.expected_period_s = expected_period_s

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return None
        ordered = sorted(values)
        index = round((len(ordered) - 1) * percentile)
        return ordered[index]

    def record_transport_timestamp(self, timestamp):
        age_ms = timestamp_age_ms(timestamp)
        if age_ms is not None:
            self.record_transport_delay(age_ms)

    def record_transport_delay(self, delay_ms):
        if not self.enabled or delay_ms is None:
            return
        with self._lock:
            self._transport_ms.append(delay_ms)

    def tick(self, processing_s=None, transport_timestamp=None, transport_delay_ms=None):
        """Record one completed stream step and optionally its input timestamp."""
        if not self.enabled:
            return
        now = time.perf_counter()
        transport_ms = transport_delay_ms
        if transport_ms is None and transport_timestamp:
            transport_ms = timestamp_age_ms(transport_timestamp)
        with self._lock:
            if self._last_tick is not None:
                self._intervals_s.append(now - self._last_tick)
            self._last_tick = now
            self._events += 1
            if processing_s is not None:
                self._processing_ms.append(processing_s * 1000)
            if transport_ms is not None:
                self._transport_ms.append(transport_ms)
            should_report = now - self._last_report >= self.report_interval_s
        if should_report:
            self.report()

    def _take_snapshot(self):
        with self._lock:
            now = time.perf_counter()
            elapsed_s = now - self._last_report
            snapshot = {
                "elapsed_s": elapsed_s,
                "events": self._events,
                "intervals_s": self._intervals_s,
                "processing_ms": self._processing_ms,
                "transport_ms": self._transport_ms,
            }
            self._last_report = now
            self._intervals_s = []
            self._processing_ms = []
            self._transport_ms = []
            self._events = 0
        return snapshot

    def report(self, final=False):
        if not self.enabled:
            return
        snapshot = self._take_snapshot()
        intervals = snapshot["intervals_s"]
        if not intervals:
            return

        mean_interval_ms = 1000 * sum(intervals) / len(intervals)
        p95_interval_ms = 1000 * self._percentile(intervals, 0.95)
        max_interval_ms = 1000 * max(intervals)
        rate_hz = 1 / (sum(intervals) / len(intervals))
        details = [
            f"rate={rate_hz:.2f}Hz",
            f"cadence(avg/p95/max)={mean_interval_ms:.1f}/{p95_interval_ms:.1f}/{max_interval_ms:.1f}ms",
        ]

        processing = snapshot["processing_ms"]
        if processing:
            details.append(
                f"processing(avg/p95)={sum(processing) / len(processing):.1f}/"
                f"{self._percentile(processing, 0.95):.1f}ms"
            )

        transport = snapshot["transport_ms"]
        if transport:
            details.append(
                f"input-age(avg/p95)={sum(transport) / len(transport):.1f}/"
                f"{self._percentile(transport, 0.95):.1f}ms"
            )

        reasons = []
        if self.expected_period_s is not None:
            expected_ms = self.expected_period_s * 1000
            details.insert(1, f"expected={expected_ms:.1f}ms")
            if self.check_cadence and mean_interval_ms > expected_ms * (1 + self.rate_tolerance):
                reasons.append(f"rate more than {self.rate_tolerance:.0%} below {1000 / expected_ms:.2f}Hz")
        if self.check_cadence and max_interval_ms > self.max_gap_ms:
            reasons.append(f"pause of {max_interval_ms:.0f}ms")
        if transport and self._percentile(transport, 0.95) > self.max_input_age_ms:
            reasons.append(f"inputs waiting more than {self.max_input_age_ms:.0f}ms")

        if self.verbose or final or reasons:
            level = f"WARNING ({', '.join(reasons)}) " if reasons else ""
            print(f"[Telemetry:{self.name}] {level}{'; '.join(details)}")

    def close(self):
        self.report(final=True)
