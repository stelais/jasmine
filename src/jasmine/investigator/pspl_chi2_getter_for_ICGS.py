import os
import csv

def main(base_dir, output_file, pspl_name = 'pspl_pars.txt'):
    results = []
    for root, dirs, files in os.walk(base_dir):
        if pspl_name in files:
            file_path = os.path.join(root, pspl_name)
            event_name = os.path.basename(os.path.dirname(root))  # e.g., event_0_917_1851

            try:
                with open(file_path, "r") as f:
                    line = f.readline().strip()
                    if line:
                        parts = line.split(",")
                        last_value = parts[-1]
                        results.append((event_name, last_value))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Sort by event name (optional)
    results.sort(key=lambda x: x[0])

    # Save results to CSV
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["event_name", "pspl_chi2_passed"])
        writer.writerows(results)

    print(f"✅ Saved summary to {output_file} with {len(results)} entries.")

if __name__ == "__main__":
    # computer_path = '/gpfsm/dnb34/sishitan/orbital_task'
    computer_path = '/Users/stela/Documents/Scripts/orbital_task'
    base_dir = f"{computer_path}/RTModel_runs/154_failures_v24_v31/second_run"
    output_file = os.path.join(base_dir, "pspl_chi2_summary.csv")
    main(base_dir, output_file)