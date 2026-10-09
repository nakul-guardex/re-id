"""Record all six RTSP cameras at the camera frame rate, without re-encoding."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

CAMERA_ENV = (
    "office_balcony",
    "office_1",
    "office_2",
    "office_3",
    "office_4",
    "office_5",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cameras = {}
    for cam_id in CAMERA_ENV:
        url = os.environ.get(f"RTSP_URL_{cam_id.upper()}", "").strip()
        if not url:
            sys.exit(f"Missing RTSP_URL_{cam_id.upper()}")
        cameras[cam_id] = url

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Recording {len(cameras)} cameras for {args.seconds} seconds into {args.out}", flush=True)

    processes = []
    for cam_id, rtsp_url in cameras.items():
        output_file = args.out / f"{cam_id}.mp4"
        log_file = args.out / f"{cam_id}.log"
        log = log_file.open("w")
        cmd = [
            "ffmpeg", "-y",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-t", str(args.seconds),
            "-c", "copy",
            "-an",
            str(output_file),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log)
        processes.append((cam_id, proc, output_file, log))
        print(f"  [{cam_id}] -> {output_file.name}", flush=True)

    failed = []
    for cam_id, proc, output_file, log in processes:
        code = proc.wait()
        log.close()
        size = output_file.stat().st_size if output_file.exists() else 0
        if code != 0 or size < 1000:
            failed.append(cam_id)
            print(f"  [{cam_id}] failed (exit {code}, {size} bytes)", flush=True)
        else:
            print(f"  [{cam_id}] saved {size / 1e6:.1f} MB", flush=True)

    if failed:
        sys.exit(f"Recording failed for: {', '.join(failed)}")
    print("All six recordings finished.", flush=True)


if __name__ == "__main__":
    main()