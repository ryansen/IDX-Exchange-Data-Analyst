import csv
import os
import time
from datetime import datetime
from pathlib import Path

import requests


BASE_URL = "https://api-trestle.corelogic.com/trestle/odata/Property"
AUTH_ENDPOINT = "https://idxexchange.com/internal-api/trestle_token.php"
AUTH_KEY_ENV = "IDXEXCHANGE_TRESTLE_TOKEN_KEY"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "csv"
TEMPLATE_FILE = OUTPUT_DIR / "CRMLSSold202604.csv"

# Missing sold months needed for the 202401-202608 combined dataset.
MONTHS_TO_EXPORT = [
    (2026, 5),
    (2026, 6),
    (2026, 7),
    (2026, 8),
]


def get_token():
    auth_key = os.environ.get(AUTH_KEY_ENV)
    if not auth_key:
        raise RuntimeError(f"Set {AUTH_KEY_ENV} in your local environment before exporting.")

    for attempt in range(1, 4):
        response = requests.get(AUTH_ENDPOINT, params={"key": auth_key}, timeout=30)
        if response.status_code == 429 and attempt < 3:
            time.sleep(60)
            continue
        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            raise RuntimeError("Error retrieving token: access_token not found")
        return token
    raise RuntimeError("Unable to retrieve token")


def get_template_fields():
    with open(TEMPLATE_FILE, newline="", encoding="cp1252", errors="replace") as file:
        return next(csv.reader(file))


def month_bounds(year, month):
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)
    return start, end


def export_month(year, month, headers, fieldnames):
    start, end = month_bounds(year, month)
    month_key = f"{year}{month:02d}"
    output_file = OUTPUT_DIR / f"CRMLSSold{month_key}.csv"

    selected_fields = list(dict.fromkeys(fieldnames))
    params = {
        "$select": ",".join(selected_fields),
        "$filter": (
            "MlsStatus eq 'Closed' and "
            "CloseDate ge "
            f"{start.isoformat(timespec='milliseconds')}Z and "
            "CloseDate lt "
            f"{end.isoformat(timespec='milliseconds')}Z"
        ),
        "$top": 1000,
    }

    total_records = 0
    request_url = BASE_URL

    with open(output_file, mode="w", newline="", encoding="cp1252", errors="replace") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        while True:
            response = requests.get(
                request_url,
                params=params,
                headers=headers,
                timeout=60,
            )
            if response.status_code != 200:
                print(f"Error exporting {month_key}: {response.status_code}")
                print(f"Error Message: {response.text}")
                break

            data = response.json()
            observations = data.get("value", [])
            writer.writerows(observations)
            total_records += len(observations)

            next_link = data.get("@odata.nextLink")
            if not next_link:
                break

            request_url = next_link
            params = None

    print(f"Total {total_records} records exported to {output_file}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = get_template_fields()
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}

    for year, month in MONTHS_TO_EXPORT:
        export_month(year, month, headers, fieldnames)


if __name__ == "__main__":
    main()
