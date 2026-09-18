"""Measure the complete raw/GF inference paths used in the main resource study."""
import argparse
import time
import numpy as np
import torch
from .common import SYSTEMS, ROOT, make_store, write
from .evaluate import load_model, forecasts, BACKBONES
from .correction import CorrectionSuite


@torch.inference_mode()
def run(method, system=None):
    model, binding = load_model(method, "joint", 17)
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("The reported GPU resource protocol requires CUDA")
    records = []
    for name in (system,) if system else SYSTEMS:
        store = make_store(name, "dev", "joint")
        for size in (1, 8):
            contexts = np.stack([store.get(i)[0] for i in range(size)]).astype(
                np.float32
            )
            device_context = (
                torch.from_numpy(contexts.copy()).transpose(1, 2).to(device)
            )
            for variant in ("raw", "full") if method in BACKBONES else ("raw",):
                memory = CorrectionSuite(("full",)) if variant == "full" else None
                serial = 0

                def invoke():
                    nonlocal serial
                    serial += 1
                    raw, back = forecasts(
                        method,
                        model,
                        contexts,
                        store,
                        binding,
                        "joint",
                        serial,
                        split="dev",
                        historical=memory is not None,
                        label="efficiency",
                        device_context=device_context,
                    )
                    return (
                        raw
                        if memory is None
                        else memory.predict(
                            device_context.transpose(1, 2).cpu().numpy(),
                            store.query_indices,
                            raw,
                            back,
                        )[0]["full"]
                    )

                if memory:
                    for _ in range(int(np.ceil(16 / size))):
                        invoke()
                for _ in range(5):
                    invoke()
                torch.cuda.synchronize(device)
                resident = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                seconds = []
                for _ in range(20):
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    invoke()
                    torch.cuda.synchronize(device)
                    seconds.append(time.perf_counter() - start)
                records.append(
                    dict(
                        method=method,
                        variant=variant,
                        system=name,
                        batch_size=size,
                        seconds=seconds,
                        median_ms=float(np.median(seconds) * 1000),
                        mean_ms=float(np.mean(seconds) * 1000),
                        p95_ms=float(np.quantile(seconds, 0.95) * 1000),
                        windows_per_second=float(size / np.mean(seconds)),
                        resident_allocated_mib=resident / 2**20,
                        peak_allocated_mib=torch.cuda.max_memory_allocated(device)
                        / 2**20,
                        incremental_peak_mib=(
                            torch.cuda.max_memory_allocated(device) - resident
                        )
                        / 2**20,
                    )
                )
    write(
        ROOT / "efficiency" / f"{method}.json",
        dict(
            checkpoint=binding,
            records=records,
            gpu=torch.cuda.get_device_name(device),
            torch_version=torch.__version__,
            loading_and_host_to_device_excluded=True,
            cpu_correction_and_prediction_transfer_included=True,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=(*BACKBONES, "tefn", "tggc"), required=True)
    p.add_argument("--system", choices=SYSTEMS)
    run(**vars(p.parse_args()))
