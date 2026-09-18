"""Immutable public engineering-system preprocessing and window access."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


SI_ZERO = (0, 0, 0, 0, 0, 0, 0)
SI_PRESSURE = (1, -1, -2, 0, 0, 0, 0)
SI_POWER = (1, 2, -3, 0, 0, 0, 0)
SI_ENERGY = (1, 2, -2, 0, 0, 0, 0)
SI_FLOW = (0, 3, -1, 0, 0, 0, 0)
SI_TEMPERATURE = (0, 0, 0, 0, 1, 0, 0)
SI_VELOCITY = (0, 1, -1, 0, 0, 0, 0)
SI_LENGTH = (0, 1, 0, 0, 0, 0, 0)
SI_FREQUENCY = (0, 0, -1, 0, 0, 0, 0)
SI_CURRENT = (0, 0, 0, 1, 0, 0, 0)
SI_VOLTAGE = (1, 2, -3, -1, 0, 0, 0)
SI_TORQUE = (1, 2, -2, 0, 0, 0, 0)
SI_TIME = (0, 0, 1, 0, 0, 0, 0)


@dataclass(frozen=True)
class Channel:
    name: str
    quantity: str
    declared_unit: str
    dimension: tuple[int, int, int, int, int, int, int]
    scale_to_si: float = 1.0
    offset_to_si: float = 0.0
    query: bool = True


@dataclass(frozen=True)
class ProcessedSystem:
    system_id: str
    values: np.ndarray
    segment_ids: np.ndarray
    split_codes: np.ndarray
    channels: tuple[Channel, ...]
    sampling_interval_seconds: float
    source_artifacts: tuple[str, ...]
    notes: tuple[str, ...] = ()


HYDRAULIC_CHANNELS = (
    *(Channel(f"PS{i}", "pressure", "bar", SI_PRESSURE, 1e5) for i in range(1, 7)),
    Channel("EPS1", "motor power", "W", SI_POWER),
    Channel("FS1", "volume flow", "L/min", SI_FLOW, 1e-3 / 60.0),
    Channel("FS2", "volume flow", "L/min", SI_FLOW, 1e-3 / 60.0),
    *(
        Channel(f"TS{i}", "temperature", "degree Celsius", SI_TEMPERATURE, 1.0, 273.15)
        for i in range(1, 5)
    ),
    Channel("VS1", "vibration velocity", "mm/s", SI_VELOCITY, 1e-3),
    Channel("CE", "cooling efficiency", "%", SI_ZERO, 0.01),
    Channel("CP", "cooling power", "kW", SI_POWER, 1000.0),
    Channel("SE", "efficiency factor", "%", SI_ZERO, 0.01),
)

METROPT_CHANNELS = (
    *(
        Channel(name, "pressure", "bar", SI_PRESSURE, 1e5)
        for name in ("TP2", "TP3", "H1", "DV_pressure", "Reservoirs")
    ),
    Channel(
        "Oil_temperature",
        "oil temperature",
        "degree Celsius",
        SI_TEMPERATURE,
        1.0,
        273.15,
    ),
    Channel("Motor_current", "motor current", "A", SI_CURRENT),
    *(
        Channel(name, "binary sensor state", "1", SI_ZERO, query=False)
        for name in (
            "COMP",
            "DV_eletric",
            "Towers",
            "MPG",
            "LPS",
            "Pressure_switch",
            "Oil_level",
        )
    ),
)

PMSM_CHANNELS = (
    Channel("u_q", "q-axis voltage", "V", SI_VOLTAGE),
    Channel(
        "coolant", "coolant temperature", "degree Celsius", SI_TEMPERATURE, 1.0, 273.15
    ),
    Channel(
        "stator_winding",
        "stator winding temperature",
        "degree Celsius",
        SI_TEMPERATURE,
        1.0,
        273.15,
    ),
    Channel("u_d", "d-axis voltage", "V", SI_VOLTAGE),
    Channel(
        "stator_tooth",
        "stator tooth temperature",
        "degree Celsius",
        SI_TEMPERATURE,
        1.0,
        273.15,
    ),
    Channel(
        "motor_speed", "motor angular speed", "rpm", SI_FREQUENCY, 2.0 * math.pi / 60.0
    ),
    Channel("i_d", "d-axis current", "A", SI_CURRENT),
    Channel("i_q", "q-axis current", "A", SI_CURRENT),
    Channel(
        "pm",
        "permanent-magnet temperature",
        "degree Celsius",
        SI_TEMPERATURE,
        1.0,
        273.15,
    ),
    Channel(
        "stator_yoke",
        "stator yoke temperature",
        "degree Celsius",
        SI_TEMPERATURE,
        1.0,
        273.15,
    ),
    Channel(
        "ambient", "ambient temperature", "degree Celsius", SI_TEMPERATURE, 1.0, 273.15
    ),
    Channel("torque", "shaft torque", "N m", SI_TORQUE),
)

STEEL_CHANNELS = (
    Channel("Usage_kWh", "active energy consumption", "kWh", SI_ENERGY, 3.6e6),
    Channel(
        "Lagging_Current_Reactive.Power_kVarh",
        "lagging reactive energy",
        "kvarh",
        SI_ENERGY,
        3.6e6,
    ),
    Channel(
        "Leading_Current_Reactive_Power_kVarh",
        "leading reactive energy",
        "kvarh",
        SI_ENERGY,
        3.6e6,
    ),
    Channel("Lagging_Current_Power_Factor", "lagging power factor", "%", SI_ZERO, 0.01),
    Channel("Leading_Current_Power_Factor", "leading power factor", "%", SI_ZERO, 0.01),
    Channel("NSM", "seconds from midnight", "s", SI_TIME, query=False),
)

BUILDING_CHANNELS = (
    Channel("Appliances", "appliance energy", "Wh", SI_ENERGY, 3600.0),
    Channel("lights", "lighting energy", "Wh", SI_ENERGY, 3600.0),
    *(
        Channel(
            f"T{i}",
            f"room {i} temperature",
            "degree Celsius",
            SI_TEMPERATURE,
            1.0,
            273.15,
        )
        for i in range(1, 10)
    ),
    *(
        Channel(f"RH_{i}", f"room {i} relative humidity", "%", SI_ZERO, 0.01)
        for i in range(1, 10)
    ),
    Channel(
        "T_out", "outdoor temperature", "degree Celsius", SI_TEMPERATURE, 1.0, 273.15
    ),
    Channel("Press_mm_hg", "atmospheric pressure", "mmHg", SI_PRESSURE, 133.322387415),
    Channel("RH_out", "outdoor relative humidity", "%", SI_ZERO, 0.01),
    Channel("Windspeed", "wind speed", "m/s", SI_VELOCITY),
    Channel("Visibility", "visibility", "km", SI_LENGTH, 1000.0),
    Channel(
        "Tdewpoint",
        "dew-point temperature",
        "degree Celsius",
        SI_TEMPERATURE,
        1.0,
        273.15,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_zip_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        member = next(
            name for name in archive.namelist() if name.lower().endswith(".csv")
        )
        with archive.open(member) as stream:
            return pd.read_csv(stream)


def _profile_split(profile: object, seed: int = 20260812) -> int:
    digest = hashlib.sha256(f"{seed}:{profile}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return 0 if fraction < 0.70 else (1 if fraction < 0.85 else 2)


def load_hydraulic(raw: Path) -> ProcessedSystem:
    # SciPy is needed only while constructing the immutable hydraulic artifact.
    # Keeping it local lets training/evaluation environments read the processed
    # NumPy arrays without inheriting a preprocessing-only binary dependency.
    from scipy import signal

    archive_path = raw / "hydraulic.zip"
    values = np.empty((2205, 120, len(HYDRAULIC_CHANNELS)), dtype=np.float32)
    native_rate = {
        **{f"PS{i}": 100 for i in range(1, 7)},
        "EPS1": 100,
        "FS1": 10,
        "FS2": 10,
        **{f"TS{i}": 1 for i in range(1, 5)},
        "VS1": 1,
        "CE": 1,
        "CP": 1,
        "SE": 1,
    }
    with zipfile.ZipFile(archive_path) as archive:
        for channel_index, channel in enumerate(HYDRAULIC_CHANNELS):
            with archive.open(f"{channel.name}.txt") as stream:
                matrix = np.loadtxt(stream, dtype=np.float32, delimiter="\t")
            if matrix.shape[0] != 2205:
                raise ValueError(
                    f"Unexpected cycle count for {channel.name}: {matrix.shape}"
                )
            rate = native_rate[channel.name]
            if rate > 2:
                resized = signal.resample_poly(matrix, up=2, down=rate, axis=1)
            elif rate == 1:
                original_t = np.arange(matrix.shape[1], dtype=np.float32)
                target_t = np.arange(120, dtype=np.float32) / 2.0
                resized = np.stack(
                    [np.interp(target_t, original_t, row) for row in matrix], axis=0
                )
            else:
                resized = matrix
            if resized.shape != (2205, 120):
                raise ValueError(
                    f"Unexpected resample for {channel.name}: {resized.shape}"
                )
            values[:, :, channel_index] = resized.astype(np.float32, copy=False)

    flat = values.reshape(-1, values.shape[-1])
    segments = np.repeat(np.arange(2205, dtype=np.int32), 120)
    cycle_split = np.full(2205, 2, dtype=np.int8)
    cycle_split[: int(0.60 * 2205)] = 0
    cycle_split[int(0.60 * 2205) : int(0.80 * 2205)] = 1
    split = np.repeat(cycle_split, 120)
    return ProcessedSystem(
        "hydraulic",
        flat,
        segments,
        split,
        HYDRAULIC_CHANNELS,
        0.5,
        (f"hydraulic.zip:{sha256_file(archive_path)}",),
        (
            "100-Hz and 10-Hz signals anti-aliased to 2 Hz; 1-Hz signals linearly interpolated",
        ),
    )


def load_metropt(raw: Path) -> ProcessedSystem:
    archive_path = raw / "metropt.zip"
    frame = _read_zip_csv(archive_path)
    times = pd.to_datetime(frame["timestamp"], errors="raise")
    delta = times.diff().dt.total_seconds().fillna(10.0).to_numpy()
    boundaries = (delta <= 0.0) | (delta > 15.0)
    segments = np.cumsum(boundaries).astype(np.int32)
    split = np.full(len(frame), -1, dtype=np.int8)
    split[times.dt.month.eq(2).to_numpy()] = 0
    split[times.dt.month.eq(3).to_numpy()] = 1
    split[times.dt.month.isin([4, 5, 6, 7, 8]).to_numpy()] = 2
    values = frame[[channel.name for channel in METROPT_CHANNELS]].to_numpy(np.float32)
    return ProcessedSystem(
        "metropt",
        values,
        segments,
        split,
        METROPT_CHANNELS,
        10.0,
        (f"metropt.zip:{sha256_file(archive_path)}",),
        (
            "Rows with 9--15 second timestamp jitter retain release order; gaps above 15 seconds split segments",
        ),
    )


def load_pmsm(raw: Path) -> ProcessedSystem:
    path = next((raw / "kagglehub").rglob("measures_v2.csv"))
    frame = pd.read_csv(path)
    profiles = frame["profile_id"].to_numpy()
    boundaries = np.r_[True, profiles[1:] != profiles[:-1]]
    segments = np.cumsum(boundaries).astype(np.int32) - 1
    profile_to_split = {item: _profile_split(item) for item in np.unique(profiles)}
    split = np.fromiter((profile_to_split[item] for item in profiles), dtype=np.int8)
    values = frame[[channel.name for channel in PMSM_CHANNELS]].to_numpy(np.float32)
    return ProcessedSystem(
        "pmsm",
        values,
        segments,
        split,
        PMSM_CHANNELS,
        0.5,
        (f"measures_v2.csv:{sha256_file(path)}",),
        ("Complete profile_id sessions assigned by SHA-256 with seed 20260812",),
    )


def load_steel(raw: Path) -> ProcessedSystem:
    archive_path = raw / "steel_industry.zip"
    frame = _read_zip_csv(archive_path)
    if len(frame) != 365 * 96:
        raise ValueError(
            "Steel series does not contain 96 records for every day of 2018"
        )
    nsm = frame["NSM"].to_numpy()
    expected_nsm = ((np.arange(len(frame)) + 1) % 96) * 900
    if not np.array_equal(nsm, expected_nsm):
        raise ValueError(
            "Steel NSM sequence is inconsistent with 15-minute release order"
        )
    split = np.full(len(frame), 2, dtype=np.int8)
    split[: int(0.60 * len(frame))] = 0
    split[int(0.60 * len(frame)) : int(0.80 * len(frame))] = 1
    values = frame[[channel.name for channel in STEEL_CHANNELS]].to_numpy(np.float32)
    return ProcessedSystem(
        "steel_industry",
        values,
        np.zeros(len(frame), dtype=np.int32),
        split,
        STEEL_CHANNELS,
        900.0,
        (f"steel_industry.zip:{sha256_file(archive_path)}",),
        (
            "Strict 15-minute axis reconstructed from release order and verified NSM slots",
            "CO2 excluded because public metadata conflict between tCO2 and ppm",
        ),
    )


def load_building(raw: Path) -> ProcessedSystem:
    archive_path = raw / "building_energy.zip"
    frame = _read_zip_csv(archive_path)
    times = pd.to_datetime(frame["date"], errors="raise")
    delta = times.diff().dt.total_seconds().dropna().to_numpy()
    if not np.all(delta == 600.0):
        raise ValueError("Building series is not strictly sampled every 10 minutes")
    split = np.full(len(frame), 2, dtype=np.int8)
    split[: int(0.60 * len(frame))] = 0
    split[int(0.60 * len(frame)) : int(0.80 * len(frame))] = 1
    values = frame[[channel.name for channel in BUILDING_CHANNELS]].to_numpy(np.float32)
    return ProcessedSystem(
        "building_energy",
        values,
        np.zeros(len(frame), dtype=np.int32),
        split,
        BUILDING_CHANNELS,
        600.0,
        (f"building_energy.zip:{sha256_file(archive_path)}",),
        (
            "rv1 and rv2 removed because the dataset authors define them as random variables",
        ),
    )


LOADERS = (load_hydraulic, load_metropt, load_pmsm, load_steel, load_building)


def _window_starts(
    segment_ids: np.ndarray,
    split_codes: np.ndarray,
    split_code: int,
    total_length: int,
    stride: int,
) -> np.ndarray:
    starts: list[np.ndarray] = []
    boundaries = np.r_[
        0, np.flatnonzero(segment_ids[1:] != segment_ids[:-1]) + 1, len(segment_ids)
    ]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        cursor = left
        while cursor < right:
            code = split_codes[cursor]
            block_end = cursor + 1
            while block_end < right and split_codes[block_end] == code:
                block_end += 1
            if code == split_code and block_end - cursor >= total_length:
                starts.append(
                    np.arange(
                        cursor, block_end - total_length + 1, stride, dtype=np.int64
                    )
                )
            cursor = block_end
    return np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)


def write_processed(
    system: ProcessedSystem,
    output_root: Path,
    context_length: int = 96,
    horizon: int = 24,
) -> dict:
    target = output_root / system.system_id
    target.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(system.values).all():
        raise ValueError(f"{system.system_id} contains non-finite values")
    np.save(target / "values.npy", system.values.astype(np.float32, copy=False))
    np.save(target / "segment_ids.npy", system.segment_ids.astype(np.int32, copy=False))
    np.save(target / "split_codes.npy", system.split_codes.astype(np.int8, copy=False))
    total = context_length + horizon
    counts = {}
    for name, code, stride in (
        ("train", 0, 4),
        ("dev", 1, horizon),
        ("test", 2, horizon),
    ):
        starts = _window_starts(
            system.segment_ids, system.split_codes, code, total, stride
        )
        if len(starts) == 0:
            raise ValueError(f"{system.system_id} has no {name} windows")
        np.save(target / f"windows_{name}.npy", starts)
        counts[name] = int(len(starts))
        if name != "train" and len(starts) > 1:
            prior_end = starts[:-1] + total
            next_target = starts[1:] + context_length
            same_segment = (
                system.segment_ids[starts[:-1]] == system.segment_ids[starts[1:]]
            )
            if np.any(same_segment & (next_target < prior_end)):
                raise ValueError(f"{system.system_id} has overlapping {name} targets")
    metadata = {
        "protocol_version": "gf-cs-2026-08-12-v3",
        "system_id": system.system_id,
        "rows": int(system.values.shape[0]),
        "channels": [asdict(channel) for channel in system.channels],
        "query_indices": [
            i for i, channel in enumerate(system.channels) if channel.query
        ],
        "sampling_interval_seconds": system.sampling_interval_seconds,
        "context_length": context_length,
        "horizon": horizon,
        "window_counts": counts,
        "split_row_counts": {
            name: int(np.sum(system.split_codes == code))
            for name, code in (("train", 0), ("dev", 1), ("test", 2), ("excluded", -1))
        },
        "segments": int(np.unique(system.segment_ids).size),
        "source_artifacts": list(system.source_artifacts),
        "notes": list(system.notes),
    }
    metadata_path = target / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    hashes = {}
    for path in sorted(target.iterdir()):
        if path.is_file():
            hashes[path.name] = sha256_file(path)
    integrity = {"files": hashes}
    (target / "integrity.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    return metadata


def preprocess_all(raw_root: Path, output_root: Path) -> list[dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = [write_processed(loader(raw_root), output_root) for loader in LOADERS]
    (output_root / "corpus_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


class WindowStore:
    """Memory-mapped access to one frozen processed system."""

    def __init__(self, root: Path, split: str) -> None:
        self.root = root
        self.split = split
        self.metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        self.values = np.load(root / "values.npy", mmap_mode="r")
        self.starts = np.load(root / f"windows_{split}.npy", mmap_mode="r")
        self.context_length = int(self.metadata["context_length"])
        self.horizon = int(self.metadata["horizon"])
        self.query_indices = np.asarray(self.metadata["query_indices"], dtype=np.int64)
        channels = self.metadata["channels"]
        self.dimensions = np.asarray(
            [item["dimension"] for item in channels], dtype=np.float32
        )
        self.unit_scale = np.asarray(
            [item["scale_to_si"] for item in channels], dtype=np.float32
        )
        self.unit_offset = np.asarray(
            [item["offset_to_si"] for item in channels], dtype=np.float32
        )
        self.sampling_interval = np.full(
            len(channels), self.metadata["sampling_interval_seconds"], dtype=np.float32
        )

    def __len__(self) -> int:
        return int(len(self.starts))

    def get(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(self.starts[index])
        cut = start + self.context_length
        end = cut + self.horizon
        return np.asarray(self.values[start:cut]).T, np.asarray(self.values[cut:end]).T

    def block_ids(
        self, indices: np.ndarray, maximum_block_seconds: float = 86400.0
    ) -> np.ndarray:
        """Return deterministic acquisition/session-aware statistical blocks."""

        selected_starts = np.asarray(self.starts[np.asarray(indices, dtype=np.int64)])
        segment_ids = np.load(self.root / "segment_ids.npy", mmap_mode="r")
        return block_ids_for_window_starts(
            selected_starts,
            np.asarray(segment_ids),
            self.context_length,
            float(self.metadata["sampling_interval_seconds"]),
            maximum_block_seconds,
            str(self.metadata["system_id"]),
        )


class QuerySubsetStore:
    """Delegate a frozen store while replacing only supervised query indices."""

    def __init__(self, base: WindowStore, query_indices: Sequence[int]) -> None:
        self.base = base
        query = np.asarray(query_indices, dtype=np.int64)
        if query.ndim != 1 or not len(query) or len(np.unique(query)) != len(query):
            raise ValueError("query subset must be a non-empty unique vector")
        if np.any(query < 0) or np.any(query >= len(base.metadata["channels"])):
            raise ValueError("query subset contains an invalid channel index")
        eligible = set(map(int, base.query_indices))
        if not set(map(int, query)).issubset(eligible):
            raise ValueError("query subset must use eligible target channels")
        self.query_indices = query
        self.root = base.root
        self.split = base.split
        self.metadata = copy_metadata = json.loads(json.dumps(base.metadata))
        copy_metadata["query_indices"] = query.tolist()
        self.values = base.values
        self.starts = base.starts
        self.context_length = base.context_length
        self.horizon = base.horizon
        self.dimensions = base.dimensions
        self.unit_scale = base.unit_scale
        self.unit_offset = base.unit_offset
        self.sampling_interval = base.sampling_interval

    def __len__(self) -> int:
        return len(self.base)

    def get(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.base.get(index)

    def block_ids(
        self, indices: np.ndarray, maximum_block_seconds: float = 86400.0
    ) -> np.ndarray:
        return self.base.block_ids(indices, maximum_block_seconds)


def load_query_partition(
    manifest_path: Path,
    partition: str,
    expected_protocol_version: str | None = None,
) -> dict[str, list[int]]:
    """Load one frozen metadata-only query partition from a protocol manifest."""

    fields = {
        "supervised_forecast": "supervised_forecast_indices",
        "unseen_target": "unseen_target_indices",
    }
    if partition not in fields:
        raise ValueError(f"unknown query partition: {partition}")
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (
        expected_protocol_version is not None
        and payload.get("protocol_version") != expected_protocol_version
    ):
        raise ValueError("query-partition manifest protocol version differs")
    if payload.get("selection_uses_values") is not False:
        raise ValueError("query partition must be selected without data values")
    if payload.get("selection_uses_test_metrics") is not False:
        raise ValueError("query partition must be selected without test metrics")
    systems = payload.get("systems")
    if not isinstance(systems, dict) or not systems:
        raise ValueError("query partition manifest has no systems")
    field = fields[partition]
    result: dict[str, list[int]] = {}
    for system_id, record in systems.items():
        if field not in record:
            raise ValueError(f"missing {field} for {system_id}")
        indices = [int(index) for index in record[field]]
        if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
            raise ValueError(f"invalid {field} for {system_id}")
        result[str(system_id)] = indices
    return result


def validate_recorded_query_partition(
    recorded: dict[str, Sequence[int]] | None,
    expected: dict[str, Sequence[int]],
    system_ids: Sequence[str],
) -> None:
    """Reject an evaluation if its checkpoint used another supervised set."""

    if recorded is None:
        raise ValueError(
            "checkpoint has no recorded supervised-query partition; it cannot "
            "support an unseen-target claim"
        )
    for system_id in system_ids:
        if system_id not in recorded or system_id not in expected:
            raise ValueError(f"query partition is missing {system_id}")
        actual = [int(index) for index in recorded[system_id]]
        wanted = [int(index) for index in expected[system_id]]
        if actual != wanted:
            raise ValueError(
                f"checkpoint supervised-query partition differs for {system_id}"
            )


def block_ids_for_window_starts(
    starts: np.ndarray,
    segment_ids: np.ndarray,
    context_length: int,
    sampling_interval_seconds: float,
    maximum_block_seconds: float,
    system_id: str,
) -> np.ndarray:
    """Group targets by acquisition segment and at most one physical day."""

    starts = np.asarray(starts, dtype=np.int64)
    segment_ids = np.asarray(segment_ids)
    if np.any(starts < 0) or np.any(starts + context_length >= len(segment_ids)):
        raise ValueError("window start falls outside the segment array")
    if sampling_interval_seconds <= 0 or maximum_block_seconds <= 0:
        raise ValueError("sampling interval and block duration must be positive")
    segment_at_start = segment_ids[starts]
    segment_origins = {
        int(segment): int(np.flatnonzero(segment_ids == segment)[0])
        for segment in np.unique(segment_at_start)
    }
    maximum_steps = max(1, round(maximum_block_seconds / sampling_interval_seconds))
    target_rows = starts + int(context_length)
    identifiers = [
        f"{system_id}:segment-{int(segment)}:day-{int((target - segment_origins[int(segment)]) // maximum_steps)}"
        for target, segment in zip(target_rows, segment_at_start)
    ]
    return np.asarray(identifiers, dtype=object)
