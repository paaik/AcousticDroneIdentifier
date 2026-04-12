#!/usr/bin/env python3
"""
Low-overhead Raspberry Pi capture tool for the FPGA's pi_sd stream.

This tool is intentionally simple:
  - uses the kernel SPI driver to clock bytes into userspace
  - does not decode frames by default
  - writes raw bytes directly to stdout or a file

Important:
  This assumes the Raspberry Pi sees the FPGA bitstream through an SPI device
  that is already configured for the board's external clocking arrangement.
  Python GPIO polling is not suitable for a 12 MHz serial stream.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import BinaryIO, Optional, Sequence

try:
    import spidev
except ImportError:
    spidev = None

try:
    import gpiod
except ImportError:
    gpiod = None


DEFAULT_CHUNK_BYTES = 8192
DEFAULT_SPI_HZ = 12_000_000


class SpiByteSource:
    def __init__(self, bus: int, device: int, chunk_size: int, max_speed_hz: int) -> None:
        if spidev is None:
            raise RuntimeError("spidev is not installed. Try: pip install spidev")

        self.chunk_size = chunk_size
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.mode = 0
        self.spi.bits_per_word = 8
        self.spi.max_speed_hz = max_speed_hz

    def read(self) -> bytes:
        return bytes(self.spi.readbytes(self.chunk_size))

    def close(self) -> None:
        self.spi.close()


def pulse_alignment_gpio(chip: str, line_offset: int, low_ms: float) -> None:
    if gpiod is None:
        raise RuntimeError("gpiod is not installed. Try: pip install gpiod")

    chip_handle = gpiod.Chip(chip)
    line = chip_handle.get_line(line_offset)
    line.request(consumer="pi-sd-dump-align", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[1])
    try:
        line.set_value(1)
        time.sleep(0.005)
        line.set_value(0)
        time.sleep(low_ms / 1000.0)
    finally:
        line.release()


def open_output(path: Optional[str]) -> BinaryIO:
    if path is None or path == "-":
        return sys.stdout.buffer
    return open(path, "wb", buffering=0)


def hexdump_line(offset: int, data: bytes) -> str:
    hex_part = " ".join(f"{byte:02x}" for byte in data)
    return f"{offset:08x}  {hex_part}"


def dump_stream(
    byte_source: SpiByteSource,
    output: BinaryIO,
    chunk_size: int,
    limit_bytes: int,
    hexdump: bool,
    stats_every: float,
) -> int:
    total = 0
    window_bytes = 0
    last_stats = time.monotonic()
    stop = False

    def handle_sigint(_sig: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sigint)

    while not stop:
        if limit_bytes > 0:
            remaining = limit_bytes - total
            if remaining <= 0:
                break
            read_len = min(chunk_size, remaining)
            byte_source.chunk_size = read_len

        chunk = byte_source.read()
        if not chunk:
            break

        if hexdump:
            for idx in range(0, len(chunk), 16):
                line = hexdump_line(total + idx, chunk[idx:idx + 16])
                output.write((line + "\n").encode("ascii"))
        else:
            output.write(chunk)

        total += len(chunk)
        window_bytes += len(chunk)

        if stats_every > 0:
            now = time.monotonic()
            if now - last_stats >= stats_every:
                rate = window_bytes / max(now - last_stats, 1e-9)
                print(
                    f"captured={total} bytes last_window_rate={rate:.0f} B/s",
                    file=sys.stderr,
                    flush=True,
                )
                window_bytes = 0
                last_stats = now

    return total


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump the FPGA pi_sd serial stream through a Raspberry Pi SPI device."
    )
    parser.add_argument("--spi-bus", type=int, default=0, help="spidev bus number.")
    parser.add_argument("--spi-device", type=int, default=0, help="spidev chip-select/device.")
    parser.add_argument(
        "--spi-hz",
        type=int,
        default=DEFAULT_SPI_HZ,
        help="SPI max speed to request from spidev.",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=DEFAULT_CHUNK_BYTES,
        help="Read size per spidev transfer.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output file path, or '-' for stdout.",
    )
    parser.add_argument(
        "--limit-bytes",
        type=int,
        default=0,
        help="Optional capture limit. 0 means run until Ctrl-C.",
    )
    parser.add_argument(
        "--hexdump",
        action="store_true",
        help="Print hex lines instead of raw bytes.",
    )
    parser.add_argument(
        "--stats-every",
        type=float,
        default=0.0,
        help="Optional stats interval in seconds written to stderr.",
    )
    parser.add_argument(
        "--align-gpio-chip",
        default=None,
        help="Optional gpiochip path/name for pulsing pi_aln before capture.",
    )
    parser.add_argument(
        "--align-gpio-line",
        type=int,
        default=None,
        help="Optional GPIO line offset used for pi_aln.",
    )
    parser.add_argument(
        "--align-low-ms",
        type=float,
        default=1.0,
        help="How long to hold pi_aln low when pulsing align.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    if args.align_gpio_chip is not None or args.align_gpio_line is not None:
        if args.align_gpio_chip is None or args.align_gpio_line is None:
            raise SystemExit("Both --align-gpio-chip and --align-gpio-line are required together.")
        pulse_alignment_gpio(args.align_gpio_chip, args.align_gpio_line, args.align_low_ms)

    byte_source = SpiByteSource(
        bus=args.spi_bus,
        device=args.spi_device,
        chunk_size=args.chunk_bytes,
        max_speed_hz=args.spi_hz,
    )
    output = open_output(args.output)

    try:
        dump_stream(
            byte_source=byte_source,
            output=output,
            chunk_size=args.chunk_bytes,
            limit_bytes=args.limit_bytes,
            hexdump=args.hexdump,
            stats_every=args.stats_every,
        )
    finally:
        byte_source.close()
        if output is not sys.stdout.buffer:
            output.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
