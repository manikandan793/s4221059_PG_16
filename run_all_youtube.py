import subprocess
import sys
import time


SCRIPTS = [
    ("youtube_to_csv.py",    "STEP 1b — JSON → CSV"),
    ("preprocessing.py",      "STEP 2  — PREPROCESSING"),
    ("network_analysis.py",   "STEP 3  — NETWORK ANALYSIS"),
    ("nlp_analysis.py",       "STEP 4  — NLP ANALYSIS"),
    ("integrated_results.py", "STEP 5  — INTEGRATED RESULTS"),
]


def main():
    total_start = time.time()
    for script, label in SCRIPTS:

        print(f"  {label}")

        start  = time.time()
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\n  FAILED at {label}")
            sys.exit(1)
        print(f"\n  {label} done ({time.time() - start:.1f}s)")


    print(f"  PIPELINE COMPLETE ({(time.time() - total_start)/60:.1f} min)")

    print("  Data    : data/")
    print("  Figures : outputs/")


if __name__ == "__main__":
    main()
