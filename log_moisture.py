import serial
import csv
import os
import sys
import re
import json
from datetime import datetime

LOG_FILE = "moisture_log.csv"
CALIBRATION_FILE = "calibration.json"


def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            return json.load(f)
    return None


def normalize_score(raw, dry_raw, wet_raw):
    score = (raw - dry_raw) / (wet_raw - dry_raw) * 10
    return max(0.0, min(10.0, score))


def classify(score):
    return "HIGH" if score >= 5.0 else "LOW"


def parse_raw(line):
    match = re.search(r"Raw:\s*(\d+)", line)
    if match:
        return int(match.group(1))
    if line.strip().isdigit():
        return int(line.strip())
    return None


def calibrate():
    if not os.path.exists(LOG_FILE):
        print("No log file yet. Run live logging first.")
        return
    groups = {}
    with open(LOG_FILE) as f:
        for row in csv.DictReader(f):
            try:
                soil = row["soil"]
                raw = int(float(row["raw"]))
            except (KeyError, ValueError):
                continue
            groups.setdefault(soil, []).append(raw)
    if not groups:
        print("No readings in the log.")
        return
    averages = {}
    for soil, raws in groups.items():
        averages[soil] = sum(raws) / len(raws)
    dry = min(averages.values())
    wet = max(averages.values())
    mid = (dry + wet) / 2
    cal = {"dry_raw": round(dry), "wet_raw": round(wet), "midpoint": mid, "soils": averages}
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(cal, f, indent=2)
    print("Per-soil averages:")
    for soil, avg in sorted(averages.items(), key=lambda kv: kv[1]):
        print(f"  {soil}: {avg:.0f}")
    print(f"\nRANGE = {cal['dry_raw']} to {cal['wet_raw']}")
    print(f"MIDPOINT = {mid:.0f} (new readings above = HIGH conductivity, below = LOW)")
    print(f"Saved to {CALIBRATION_FILE}.")


def parse_args(argv):
    port = "COM7"
    soil = None
    calibrate = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--soil" and i + 1 < len(argv):
            soil = argv[i + 1]
            i += 2
        elif arg == "--calibrate":
            calibrate = True
            i += 1
        elif not arg.startswith("--"):
            port = arg
            i += 1
        else:
            i += 1
    return port, soil, calibrate


def main():
    port, soil, do_calibrate = parse_args(sys.argv[1:])
    if do_calibrate:
        calibrate()
        return

    if soil is None:
        soil = input("Soil label (e.g. soil #1): ").strip() or "unknown"

    cal = load_calibration()
    print(f"Logging moisture on {port} @ 9600 -> {LOG_FILE} (Ctrl+C to stop)")
    if cal:
        print(f"Range: raw {cal['dry_raw']}-{cal['wet_raw']}, midpoint {cal['midpoint']} (>= midpoint HIGH, < LOW)")
    else:
        print("No calibration yet. Log a few soils, then run: python log_moisture.py --calibrate")
    print()

    file_exists = os.path.exists(LOG_FILE)
    session_raws = []
    with serial.Serial(port, 9600, timeout=1) as ser, open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "soil", "raw"])
        if not file_exists:
            writer.writeheader()
        try:
            while True:
                line = ser.readline().decode("utf-8", errors="replace")
                if not line:
                    continue
                raw = parse_raw(line)
                if raw is None:
                    continue
                session_raws.append(raw)
                stamp = datetime.now().isoformat(timespec="seconds")
                writer.writerow({"timestamp": stamp, "soil": soil, "raw": raw})
                f.flush()
                if cal:
                    score = normalize_score(raw, cal["dry_raw"], cal["wet_raw"])
                    level = classify(score)
                    print(f"{stamp}  soil={soil}  raw={raw}  conductivity={score:.1f}  {level}", flush=True)
                else:
                    print(f"{stamp}  soil={soil}  raw={raw}", flush=True)
        except KeyboardInterrupt:
            print("\nLogging stopped. Data saved to", LOG_FILE)
            print_summary(soil, session_raws, cal)


def print_summary(soil, raws, cal):
    if not raws:
        return
    avg = sum(raws) / len(raws)
    print(f"\n=== Result for {soil} ===")
    print(f"Readings: {len(raws)}  Average raw = {avg:.0f}")
    if not cal:
        print("No calibration yet - run: python log_moisture.py --calibrate")
        return
    score = normalize_score(avg, cal["dry_raw"], cal["wet_raw"])
    level = classify(score)
    print(f"Conductivity score = {score:.1f} ({level})  |  midpoint = {cal['midpoint']:.0f}")
    if level == "HIGH":
        print("HIGH conductivity = moist, ion-rich soil - conditions favorable for a fungal network")
    else:
        print("LOW conductivity = drier soil - lower fungal-network likelihood")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print("ERROR:", e)
        print("Make sure the Serial Monitor in VS Code is closed and only this script uses the port.")
        sys.exit(1)
