from src.drift_detection import detect_drift

if __name__ == "__main__":
    ref = "/app/data/test.csv"
    cur = "/app/data/drifted_test.csv"

    results = detect_drift(
        reference_csv=ref,
        current_csv=cur,
        target="SOURCE",
        report_dir="/app/reports",
    )
    print("=== Drift results ===")
    for k, v in results.items():
        if k != "raw":  # avoid printing full Evidently dump
            print(f"{k}: {v}")
