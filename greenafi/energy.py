"""Energy meters.

Priority:
  1. macOS (Apple Silicon): `powermetrics` (needs sudo; run `sudo -v` first).
     Integrates "Combined Power (CPU + GPU + ANE)" over the measured window.
  2. Linux Intel/AMD: RAPL package counters in /sys/class/powercap.
  3. Neither available: no measurement; energy is *estimated* later as
     time x assumed power, and is labelled as an estimate in every output.

Every measurement subtracts the idle power measured just before, so the number
reported is the energy *attributable to the workload*.
"""
from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time


class NullMeter:
    name = "none"

    def start(self):
        self.t0 = time.perf_counter()

    def stop(self) -> dict:
        return {"seconds": time.perf_counter() - self.t0, "joules": None}


class RaplMeter:
    name = "rapl"

    def __init__(self):
        self.paths = [p for p in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj") if p.count(":") == 1]
        self.max = [int(open(p.replace("energy_uj", "max_energy_range_uj")).read()) for p in self.paths]

    @staticmethod
    def available() -> bool:
        ps = glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj")
        return bool(ps) and all(os.access(p, os.R_OK) for p in ps)

    def _read(self):
        return [int(open(p).read()) for p in self.paths]

    def start(self):
        self.e0 = self._read()
        self.t0 = time.perf_counter()

    def stop(self):
        e1 = self._read()
        dt = time.perf_counter() - self.t0
        uj = sum((b - a) % m for a, b, m in zip(self.e0, e1, self.max))
        return {"seconds": dt, "joules": uj / 1e6}


class PowermetricsMeter:
    name = "powermetrics"
    PAT = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW")
    INTERVAL_MS = 100

    @staticmethod
    def available() -> bool:
        if platform.system() != "Darwin" or not shutil.which("powermetrics"):
            return False
        return subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0

    def start(self):
        self.out = tempfile.NamedTemporaryFile(suffix=".txt", delete=False).name
        self.proc = subprocess.Popen(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power,ane_power",
             "-i", str(self.INTERVAL_MS), "-o", self.out],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(0.35)  # let the first sample start
        self.t0 = time.perf_counter()

    def stop(self):
        dt = time.perf_counter() - self.t0
        time.sleep(self.INTERVAL_MS / 1000 * 1.5)
        subprocess.run(["sudo", "-n", "pkill", "-INT", "-f", f"powermetrics.*{self.out}"], capture_output=True)
        self.proc.wait(timeout=10)
        mw = [float(v) for v in self.PAT.findall(open(self.out).read())]
        os.unlink(self.out)
        if not mw:
            return {"seconds": dt, "joules": None}
        mean_w = sum(mw) / len(mw) / 1000
        return {"seconds": dt, "joules": mean_w * dt}


def make_meter():
    if PowermetricsMeter.available():
        return PowermetricsMeter()
    if RaplMeter.available():
        return RaplMeter()
    return NullMeter()


def idle_power(meter, seconds: float = 8.0) -> float | None:
    """Average power (W) with nothing running, used as the baseline."""
    meter.start()
    time.sleep(seconds)
    r = meter.stop()
    return None if r["joules"] is None else r["joules"] / r["seconds"]
