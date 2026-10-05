"""Bounded, auditable transport pacing for native AMA frozen-prefix probes."""
from __future__ import annotations

import copy
import math
import time
from email.utils import parsedate_to_datetime
from .model import Model, ModelError


def retry_after_seconds(headers, now=None):
    value = (headers or {}).get('retry-after')
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - (time.time() if now is None else now))
        except (TypeError, ValueError, OverflowError):
            return None


class ProbeTransport:
    """One sequential stream; fail closed when bounded rate-limit recovery fails."""
    def __init__(self, settings, on_attempt, *, sleep=time.sleep, clock=time.time):
        self.delay = settings.get('request_delay_seconds', 3.0)
        self.max_attempts = settings.get('max_429_attempts', 6)
        self.max_wait = settings.get('max_retry_wait_seconds', 240.0)
        if type(self.delay) not in (int, float) or not math.isfinite(self.delay) or not 0 <= self.delay <= 60:
            raise ValueError('request delay must be 0..60 seconds')
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 6:
            raise ValueError('max_429_attempts must be 1..6')
        if type(self.max_wait) not in (int, float) or not math.isfinite(self.max_wait) or not 30 <= self.max_wait <= 600:
            raise ValueError('max_retry_wait_seconds must be 30..600')
        self.on_attempt, self.sleep, self.clock = on_attempt, sleep, clock
        self.stop_reason = None

    def complete(self, model, history, tools, job_id):
        if self.stop_reason:
            raise ModelError(self.stop_reason)
        attempts = []
        for index in range(self.max_attempts):
            self.sleep(self.delay)
            record = {'job_id': job_id, 'attempt': index + 1, 'started_at': self.clock(),
                      'status': 'started', 'usage': None}
            # Write before sending; interrupted attempts have explicitly unknown outcomes.
            self.on_attempt(copy.deepcopy(record))
            try:
                result = Model.complete(model, history, tools)
            except ModelError as exc:
                wait = retry_after_seconds(exc.rate_limit_headers, self.clock())
                if wait is None:
                    wait = min(30 * 2 ** index, self.max_wait)
                retry = exc.http_status == 429 and index + 1 < self.max_attempts and wait <= self.max_wait
                record.update(status='rejected' if exc.http_status == 429 else 'failed',
                              http_status=exc.http_status, rate_limit_headers=exc.rate_limit_headers,
                              error=str(exc), usage=copy.deepcopy(model.last_usage),
                              retry_wait_seconds=wait if retry else None)
                attempts.append(record)
                model.transport_attempts = copy.deepcopy(attempts)
                self.on_attempt(copy.deepcopy(record))
                if exc.http_status == 429:
                    self.delay = min(30.0, max(3.0, self.delay * 2))
                    if retry:
                        self.sleep(wait)
                        continue
                    self.stop_reason = 'rate_limit_recovery_exhausted; run stopped before next job'
                elif exc.http_status is None or exc.http_status >= 500:
                    # Completion/billing may be unknown; never resample silently.
                    self.stop_reason = 'ambiguous_transport_failure; run stopped before next job'
                raise
            except BaseException:
                model.transport_attempts = copy.deepcopy(attempts)
                self.stop_reason = 'interrupted_transport; run stopped before next job'
                raise
            record.update(status='completed', completed_at=self.clock(),
                          usage=copy.deepcopy(model.last_usage))
            attempts.append(record)
            model.transport_attempts = copy.deepcopy(attempts)
            self.on_attempt(copy.deepcopy(record))
            return result
        raise AssertionError('unreachable')


class PacedProbeModel(Model):
    def __init__(self, name, config, timeout, transport, job_id):
        super().__init__(name, config, timeout)
        self.transport, self.job_id = transport, job_id
        self.transport_attempts = []

    def complete(self, history, tools=None):
        return self.transport.complete(self, history, tools, self.job_id)
