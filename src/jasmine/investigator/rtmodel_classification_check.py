import os
import re
import pandas as pd


def main(directory_with_events_, master_file_, output_csv_):
    master_df = pd.read_csv(master_file_)
    master_df['event_name_'] = (
        master_df['lcname'].str.replace('OMPLDG_croin_cassan/OMPLDG_croin_cassan_', 'event_', regex=False).str.replace(
            '.det.lc', '', regex=False))

    # Prepare CSV columns
    header = ["event_name_",
              "found_a_planetary_solution",
              "found_a_2L1S_solution",
              "complete_classification",
              "models"]
    rows = []

    # Finding full classification descriptions for each event
    for event_name in os.listdir(directory_with_events_):
        event_path = os.path.join(directory_with_events_, event_name)

        # Skip non-directories and hidden files
        if not os.path.isdir(event_path) or event_name.startswith('.'):
            continue

        # DEBUG LINE
        # # try to read Nature.txt file and if not found, skip the event
        # if not os.path.isfile(os.path.join(event_path, "Nature.txt")):
        #     print(f"Nature.txt not found for event: {event_name}, skipping.")
        #     continue

        # Read Nature.txt file
        nature_file = os.path.join(event_path, "Nature.txt")
        with open(nature_file, "r") as f:
            lines = f.readlines()
            content = ''.join(lines)

        # Extract the line starting with 'Successful:'
        complete_classification = ""
        match = re.search(r"Successful:\s*(.*)", content)
        if match:
            complete_classification = match.group(1).strip()
        found_a_planet = "Planetary lens" in complete_classification # Save if planetary solution found

        # Find the start of the 'chisquare   model' section
        try:
            start = next(i for i, line in enumerate(lines) if "chisquare" in line)
        except StopIteration:
            start = None

        models = []
        if start is not None:
            for line in lines[start + 1:]:
                parts = line.split()
                if len(parts) != 2 or not parts[1].endswith(".txt"):
                    break
                models.append(parts[1].replace(".txt", ""))

        # Check if any model starts with 'L'
        found_a_2L1S_solution = any(m.startswith("L") for m in models)
        rows.append([event_name, found_a_planet, found_a_2L1S_solution, complete_classification, models])

    # Create DataFrame
    df = pd.DataFrame(rows, columns=header)
    df = df.sort_values(by='event_name_')
    # Save to CSV
    df.to_csv(output_csv_, index=False)
    print(f"Saved DataFrame to: {output_csv_}")


if __name__ == '__main__':
    # EXAMPLE OF HOW TO USE THIS SCRIPT
    # general_path = '/Users/stela/Documents/Scripts/orbital_task'
    general_path = '/gpfsm/dnb34/sishitan/orbital_task'

    # Directory containing all event folders
    directory_with_events = f"{general_path}/RTModel_runs/154_failures_v24_v31/second_run"
    # Master CSV file
    master_file = f'{general_path}/data/gulls_orbital_motion_extracted/OMPLDG_croin_cassan.sample.csv'
    # Output CSV
    output_csv = f"{directory_with_events}/154_failures_now_with_ICGS_events_classification.csv"
    main(directory_with_events_=directory_with_events, master_file_=master_file, output_csv_=output_csv)
    print("Done")
