#!/bin/bash
# Download datasets from their original publishers.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

STAGING="$DATA_ROOT/_downloads"
mkdir -p "$STAGING"

download() {
    local url=$1 output=$2
    if [ ! -s "$output" ]; then
        echo "[download_data] downloading $url"
        curl -fL --retry 5 --retry-delay 5 -C - -o "$output" "$url"
    fi
}

extract_zip() {
    local archive=$1 output=$2
    rm -rf "$output"
    mkdir -p "$output"
    unzip -q "$archive" -d "$output"
}

install_tree() {
    local source=$1 target=$2
    if [ -d "$target" ]; then
        echo "[download_data] skip $target (already exists)"
        return
    fi
    mkdir -p "$(dirname "$target")"
    mv "$source" "$target"
}

if [ ! -d "$XACLE_AUDIO_DIR" ]; then
    archive="$STAGING/XACLE_dataset_train_val.zip"
    unpacked="$STAGING/xacle_train"
    download "https://y-okamoto.sakura.ne.jp/XACLE_Challenge/2025/dataset/XACLE_dataset_train_val.zip" "$archive"
    extract_zip "$archive" "$unpacked"
    source_dir="$(find "$unpacked" -type d -path "*/wav/train" -print -quit)"
    [ -n "$source_dir" ] || { echo "XACLE train wav directory not found" >&2; exit 1; }
    install_tree "$source_dir" "$XACLE_AUDIO_DIR"
    metadata_dir="$(find "$unpacked" -type d -name meta_data -print -quit)"
    if [ -n "$metadata_dir" ] && [ ! -d "$DATA_ROOT/xacle/meta_data" ]; then
        install_tree "$metadata_dir" "$DATA_ROOT/xacle/meta_data"
    fi
    rm -rf "$unpacked"
fi

if [ ! -d "$XACLE_TEST_DIR" ] || [ ! -s "$XACLE_TEST_CSV" ]; then
    archive="$STAGING/XACLE_test_data_with_score.zip"
    unpacked="$STAGING/xacle_test"
    download "https://y-okamoto.sakura.ne.jp/XACLE_Challenge/2025/dataset/XACLE_test_data_with_score.zip" "$archive"
    extract_zip "$archive" "$unpacked"
    source_dir="$(find "$unpacked" -type d -path "*/wav/test" -print -quit)"
    if [ -z "$source_dir" ]; then
        wav_file="$(find "$unpacked" -type f -name "*.wav" -print -quit)"
        [ -n "$wav_file" ] && source_dir="$(dirname "$wav_file")"
    fi
    [ -n "$source_dir" ] || { echo "XACLE test wav directory not found" >&2; exit 1; }
    install_tree "$source_dir" "$XACLE_TEST_DIR"
    mkdir -p "$(dirname "$XACLE_TEST_CSV")"
    score_csv="$(find "$unpacked" -type f \( -name test_average.csv -o -iname "*test*score*.csv" \) -print -quit)"
    [ -n "$score_csv" ] || { echo "XACLE test score CSV not found" >&2; exit 1; }
    cp "$score_csv" "$XACLE_TEST_CSV"
    rm -rf "$unpacked"
fi

if [ ! -d "$RELATE_AUDIO_DIR" ]; then
    archive="$STAGING/RELATE_wave.zip"
    unpacked="$STAGING/relate"
    download "https://sarulab.sakura.ne.jp/kanamori/RELATE_open_dataset/RELATE_wave.zip" "$archive"
    extract_zip "$archive" "$unpacked"
    source_dir="$(find "$unpacked" -type d -name RELATE_wave -print -quit)"
    [ -n "$source_dir" ] || source_dir="$unpacked"
    install_tree "$source_dir" "$RELATE_AUDIO_DIR"
fi
mkdir -p "$(dirname "$RELATE_CSV")"
download "https://raw.githubusercontent.com/sarulab-speech/RELATE/main/scores/REL.csv" "$RELATE_CSV"

if [ ! -d "$PAM_AUDIO_ROOT" ]; then
    archive="$STAGING/pam_human_eval.zip"
    unpacked="$STAGING/pam"
    download "https://zenodo.org/records/10737388/files/human_eval.zip?download=1" "$archive"
    extract_zip "$archive" "$unpacked"
    source_dir="$(find "$unpacked" -type d -name human_eval -print -quit)"
    [ -n "$source_dir" ] || { echo "PAM human_eval directory not found" >&2; exit 1; }
    install_tree "$source_dir" "$DATA_ROOT/pam/human_eval"
    rm -rf "$unpacked"
fi

if [ ! -d "$CLOTHO_AUDIO_ROOT" ] || [ ! -s "$CLOTHO_CSV" ]; then
    archive="$STAGING/audiobertscore.zip"
    unpacked="$STAGING/audiobertscore"
    download "https://github.com/lourson1091/audiobertscore/archive/refs/heads/main.zip" "$archive"
    extract_zip "$archive" "$unpacked"
    repo_dir="$(find "$unpacked" -mindepth 1 -maxdepth 1 -type d -print -quit)"
    mkdir -p "$DATA_ROOT/clotho"
    [ -d "$CLOTHO_AUDIO_ROOT" ] || install_tree "$repo_dir/wave_all_16k" "$CLOTHO_AUDIO_ROOT"
    [ -s "$CLOTHO_CSV" ] || cp "$repo_dir/clotho_ovl_rel_test_set.csv" "$CLOTHO_CSV"
    rm -rf "$unpacked"
fi

echo "[download_data] done"
