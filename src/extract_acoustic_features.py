"""
Parse LENA .its XML files into 1-minute acoustic features per child.

Uses the consolidated per-day mapping (MAPPING_CONSOLIDATED.csv) to resolve
each .its file to a Subject_ID based on (recording_date, LENA recorder ID).
"""

import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONSOLIDATED_MAPPING = 'data/mapping/MAPPING_CONSOLIDATED.csv'

LENA_ITS_DIR = 'data/lena/'
ACOUSTIC_OUTPUT = 'acoustic_features_1min.csv'

# LENA IDs that should never be assigned to a child
PLACEHOLDER_LENA_IDS = {'22222'}

# LENA speaker tags that count as overlap
OVERLAP_COLS = ['OLF', 'OLN', 'NOF', 'NON']

# ==========================================
# 2. HELPERS
# ==========================================
def parse_duration(val):
    """Convert LENA ISO-8601 duration (e.g. 'P14.09S') to seconds."""
    try:
        if not val or pd.isna(val):
            return 0.0
        return pd.to_timedelta(val).total_seconds()
    except Exception:
        return 0.0


def load_day_mappings(mapping_file):
    """
    Load the consolidated per-day mapping.

    Each row is (Date, Subject_ID, LENA_DLP_ID, ...). Rows with missing
    Date, Subject_ID, or LENA_DLP_ID are ignored.

    Returns:
        day_lookup: {date: {lena_id: subject_id}}
        all_lena_ids: set of every LENA ID seen
    """
    df = pd.read_csv(mapping_file)
    df = df.dropna(subset=['Date', 'Subject_ID', 'LENA_DLP_ID'])

    day_lookup = {}
    all_lena_ids = set()

    for _, row in df.iterrows():
        try:
            date_obj = pd.to_datetime(row['Date']).date()
        except Exception:
            continue
        try:
            lena_id = str(int(float(row['LENA_DLP_ID'])))
        except (ValueError, TypeError):
            continue
        if lena_id in PLACEHOLDER_LENA_IDS:
            continue
        subject = str(row['Subject_ID']).strip()
        if not subject or subject.lower() == 'nan':
            continue

        day_lookup.setdefault(date_obj, {})
        day_lookup[date_obj].setdefault(lena_id, subject)  # first entry wins on dupes
        all_lena_ids.add(lena_id)

    return day_lookup, all_lena_ids


def get_its_recording_date(its_file_path):
    """Extract recording date (America/New_York) from the first <Recording> tag."""
    try:
        root = ET.parse(its_file_path).getroot()
        rec = root.find(".//Recording")
        if rec is None:
            return None
        start_str = rec.get("startClockTime")
        if not start_str:
            return None
        return pd.to_datetime(start_str, utc=True).tz_convert('America/New_York').date()
    except Exception:
        return None


def find_lena_id_in_filename(its_filename, all_lena_ids):
    """Match a LENA recorder ID against the filename (longest first)."""
    for d in sorted(all_lena_ids, key=len, reverse=True):
        if d in its_filename or d.zfill(6) in its_filename:
            return d
    return None


def resolve_subject(its_file_path, day_lookup, all_lena_ids):
    """Resolve (its_file) -> subject_id using (recording_date, LENA ID)."""
    rec_date = get_its_recording_date(its_file_path)
    if rec_date is None:
        return None, {'reason': 'no_date_in_xml', 'date': None, 'lena_id': None}

    lena_id = find_lena_id_in_filename(its_file_path.name, all_lena_ids)
    if lena_id is None:
        return None, {'reason': 'no_lena_id_in_filename', 'date': rec_date, 'lena_id': None}

    if rec_date not in day_lookup:
        return None, {'reason': 'no_mapping_for_date', 'date': rec_date, 'lena_id': lena_id}

    sid = day_lookup[rec_date].get(lena_id)
    if sid is None:
        return None, {'reason': 'lena_not_assigned_that_day', 'date': rec_date, 'lena_id': lena_id}

    return sid, {'reason': 'ok', 'date': rec_date, 'lena_id': lena_id}


# ==========================================
# 3. LENA PARSING
# ==========================================
def process_lena_its(its_file_path, subject_id):
    """Parse an .its file into per-minute acoustic features for one child."""
    root = ET.parse(its_file_path).getroot()
    records = []

    for rec in root.findall(".//Recording"):
        rec_start_str = rec.get("startClockTime")
        if not rec_start_str:
            continue

        base_time = pd.to_datetime(rec_start_str, utc=True).tz_convert('America/New_York')

        for seg in rec.findall(".//Segment"):
            spkr = seg.get("spkr", "")
            start_offset = parse_duration(seg.get("startTime"))
            end_offset = parse_duration(seg.get("endTime"))
            seg_dur = end_offset - start_offset
            if seg_dur <= 0:
                continue

            # Child utterance count
            try:
                utt_count = float(seg.get("childUttCnt", 0))
            except (ValueError, TypeError):
                utt_count = 0.0

            # Child utterance length in seconds (ISO-8601 duration, e.g. "PT0.43S")
            utt_len = parse_duration(seg.get("childUttLen", "PT0S"))

            # Adult word count (any adult)
            wc_str = seg.get("femaleAdultWordCnt",
                             seg.get("maleAdultWordCnt",
                                     seg.get("adultWordCnt", "0")))
            try:
                awc = float(wc_str)
            except Exception:
                awc = 0.0

            records.append({
                'SUBJECTID': subject_id,
                'start': base_time + pd.Timedelta(seconds=start_offset),
                'end': base_time + pd.Timedelta(seconds=end_offset),
                'awc': awc,
                'utt_count': utt_count,
                'utt_len': utt_len,
                'is_overlap': spkr in OVERLAP_COLS,
            })

    df_segments = pd.DataFrame(records)
    if df_segments.empty:
        return pd.DataFrame()

    # Distribute segment values proportionally into 1-min bins
    binned_data = []
    for _, row in df_segments.iterrows():
        seg_start, seg_end = row['start'], row['end']
        seg_dur = (seg_end - seg_start).total_seconds()

        current_bin = seg_start.floor('1Min')
        bin_end_limit = seg_end.ceil('1Min')

        while current_bin < bin_end_limit:
            next_bin = current_bin + pd.Timedelta('1Min')
            overlap_dur = (min(seg_end, next_bin) - max(seg_start, current_bin)).total_seconds()

            if overlap_dur > 0:
                ratio = overlap_dur / seg_dur
                binned_data.append({
                    'TIME_LOCAL': current_bin,
                    'SUBJECTID': row['SUBJECTID'],
                    'awc': row['awc'] * ratio,
                    'utt_count': row['utt_count'] * ratio,
                    'utt_len': row['utt_len'] * ratio,
                    'overlap_dur': overlap_dur if row['is_overlap'] else 0.0,
                })
            current_bin = next_bin

    df_binned = pd.DataFrame(binned_data)
    if df_binned.empty:
        return pd.DataFrame()

    final_df = df_binned.groupby(['TIME_LOCAL', 'SUBJECTID']).agg({
        'awc': 'sum',
        'utt_count': 'sum',
        'utt_len': 'sum',
        'overlap_dur': 'sum',
    }).reset_index()

    return final_df.rename(columns={
        'awc': 'AWC_1min_Sum',
        'utt_count': 'Child_Utt_Count_1min',
        'utt_len': 'Child_Utt_Len_1min_Sum',
        'overlap_dur': 'Overlap_1min_Sum',
    })


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    DEBUG_MODE = False

    print(f"\n{'='*60}\n   ACOUSTIC FEATURE EXTRACTION\n{'='*60}")

    # Load consolidated per-day mapping
    try:
        day_lookup, all_lena_ids = load_day_mappings(CONSOLIDATED_MAPPING)
    except Exception as e:
        print(f"CRITICAL ERROR loading {CONSOLIDATED_MAPPING}: {e}")
        exit()

    if not day_lookup:
        print(f"CRITICAL ERROR: No usable rows found in {CONSOLIDATED_MAPPING}")
        exit()

    print(f"Loaded {len(day_lookup)} day(s), "
          f"{len(all_lena_ids)} distinct LENA recorders.")

    # Find .its files
    lena_files = list(Path(LENA_ITS_DIR).rglob('*.its'))
    if DEBUG_MODE:
        lena_files = lena_files[:1]
    print(f"Found {len(lena_files)} .its files in {LENA_ITS_DIR}")

    # Resolve + process each file
    all_lena = []
    skipped = {
        'no_date_in_xml': [],
        'no_lena_id_in_filename': [],
        'no_mapping_for_date': [],
        'lena_not_assigned_that_day': [],
    }
    processed_log = []

    for its_file in tqdm(lena_files, desc="Processing LENA XMLs"):
        sid, status = resolve_subject(its_file, day_lookup, all_lena_ids)

        if sid is None:
            skipped[status['reason']].append((its_file.name, status['date'], status['lena_id']))
            continue

        processed_log.append((its_file.name, status['date'], status['lena_id'], sid))
        binned = process_lena_its(its_file, sid)
        if not binned.empty:
            all_lena.append(binned)

    # Report skipped
    total_skipped = sum(len(v) for v in skipped.values())
    if total_skipped > 0:
        print(f"\n--- SKIPPED {total_skipped} of {len(lena_files)} FILES ---")
        for reason, items in skipped.items():
            if not items:
                continue
            print(f"\n  [{reason}] {len(items)} file(s):")
            for fname, d, lid in items[:25]:
                print(f"    {fname}  (date={d}, lena_id={lid})")
            if len(items) > 25:
                print(f"    ... and {len(items) - 25} more")

    # Trace log
    if processed_log:
        log_df = pd.DataFrame(processed_log,
                              columns=['its_file', 'recording_date', 'lena_id', 'subject_id'])
        log_df.to_csv('lena_subject_assignments.csv', index=False)
        print("\n  Wrote per-file assignment trace to: lena_subject_assignments.csv")

    # Save output
    if all_lena:
        acoustic_df = pd.concat(all_lena, ignore_index=True)
        acoustic_df.to_csv(ACOUSTIC_OUTPUT, index=False)
        print(f"\n✅ Saved {len(acoustic_df)} minutes to {ACOUSTIC_OUTPUT}")
        print(f"   Subjects covered: {acoustic_df['SUBJECTID'].nunique()}")
    else:
        print("\n⚠ No acoustic data produced.")