from pathlib import Path

def resolve_metadata_csv_path(config):
    """Return the metadata CSV path with EXACT audio configuration match."""
    data_cfg = config.get("data", {})
    metadata_dir = data_cfg.get("metadata_dir")

    if not metadata_dir:
        raise FileNotFoundError("No metadata directory configured in config['data']['metadata_dir']")

    metadata_dir_path = Path(metadata_dir)
    if not metadata_dir_path.is_dir():
        raise FileNotFoundError(f"Metadata directory does not exist: {metadata_dir_path}")

    audio_cfg = config.get("audio", {})
    required_keys = ["sr", "n_fft", "hop_length", "n_mels", "segment_size"]
    missing = [k for k in required_keys if k not in audio_cfg]
    if missing:
        raise KeyError(f"Missing keys in config['audio']: {missing}")

    # Build exact expected filename signature
    signature = f"sr{audio_cfg['sr']}_nfft{audio_cfg['n_fft']}_hop{audio_cfg['hop_length']}_nmel{audio_cfg['n_mels']}_seg{audio_cfg['segment_size']}"

    # We expect the csv to CONTAIN this exact signature as a whole block
    expected_name_part = signature.lower()

    csv_files = sorted(metadata_dir_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No metadata CSV files found in {metadata_dir_path}")

    # Find exact match
    matches = [p for p in csv_files if expected_name_part in p.stem.lower()]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No metadata CSV found with EXACT signature: {signature}\n"
            f"Looked in: {metadata_dir_path}\n"
            f"Available: {[p.name for p in csv_files]}"
        )

    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple CSVs found with signature {signature}. Be explicit.\n"
            f"Matches: {[p.name for p in matches]}"
        )

    return str(matches[0])
