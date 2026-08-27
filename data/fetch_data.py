"""
Fetch the five raw dataset files this study uses and verify each one against
checksums.txt. Exits with an error rather than silently continuing if a
download doesn't match - the point is to reproduce the exact data the
experiment ran on, not "a" copy of each dataset (UCI/OpenML have occasionally
revised files under the same name in the past).

Usage (from this directory):
    python fetch_data.py
"""

import hashlib
import io
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKSUMS_FILE = os.path.join(HERE, 'checksums.txt')


def load_checksums():
    checksums = {}
    with open(CHECKSUMS_FILE, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            digest, filename = line.split(None, 1)
            checksums[filename] = digest.lower()
    return checksums


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_or_die(filename, data, checksums):
    expected = checksums.get(filename)
    if expected is None:
        raise SystemExit(f"No expected checksum recorded for {filename!r} "
                          f"in checksums.txt - refusing to write it blind.")
    actual = sha256_of(data)
    if actual != expected:
        raise SystemExit(
            f"Checksum mismatch for {filename}:\n"
            f"  expected {expected}\n"
            f"  got      {actual}\n"
            f"The upstream file has changed since this checksum was recorded, "
            f"or the download is corrupt. Not writing it - this would silently "
            f"reproduce a different experiment. See data/README.md."
        )
    return True


def fetch_zip_member(url, member_name, filename, checksums):
    """Download a zip, pull one member out of it, verify, write to data/."""
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        data = zf.read(member_name)
    verify_or_die(filename, data, checksums)
    with open(os.path.join(HERE, filename), 'wb') as out:
        out.write(data)
    print(f"  OK: {filename} ({len(data):,} bytes, checksum verified)")


def fetch_nested_zip_member(url, inner_zip_name, member_name, filename, checksums):
    """Bank Marketing's zip contains a further bank.zip with bank-full.csv inside."""
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as outer:
        inner_blob = outer.read(inner_zip_name)
    with zipfile.ZipFile(io.BytesIO(inner_blob)) as inner:
        data = inner.read(member_name)
    verify_or_die(filename, data, checksums)
    with open(os.path.join(HERE, filename), 'wb') as out:
        out.write(data)
    print(f"  OK: {filename} ({len(data):,} bytes, checksum verified)")


def fetch_plain(url, filename, checksums):
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    verify_or_die(filename, data, checksums)
    with open(os.path.join(HERE, filename), 'wb') as out:
        out.write(data)
    print(f"  OK: {filename} ({len(data):,} bytes, checksum verified)")


def main():
    checksums = load_checksums()

    fetch_zip_member(
        'https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip',
        'wdbc.data', 'wdbc.data', checksums)
    fetch_zip_member(
        'https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip',
        'wdbc.names', 'wdbc.names', checksums)
    fetch_zip_member(
        'https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip',
        'german.data', 'german.data', checksums)
    fetch_zip_member(
        'https://archive.ics.uci.edu/static/public/2/adult.zip',
        'adult.data', 'adult.data', checksums)
    fetch_nested_zip_member(
        'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip',
        'bank.zip', 'bank-full.csv', 'bank-full.csv', checksums)
    fetch_plain(
        'https://openml.org/data/v1/download/21756251/telco-customer-churn.arff',
        'dataset.arff', checksums)

    print("\nAll six files downloaded and verified against checksums.txt.")


if __name__ == '__main__':
    try:
        main()
    except zipfile.BadZipFile:
        print("A downloaded file wasn't a valid zip - the source URL has "
              "likely moved. Check the dataset pages linked in README.md and "
              "download manually, then verify by hand against checksums.txt.",
              file=sys.stderr)
        raise
