#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REQUIREMENTS_FILE="${EMOTYC_REQUIREMENTS:-$ROOT_DIR/requirements.txt}"
MODEL_DIR="${EMOTYC_MODEL_DIR:-$ROOT_DIR/model_onnx}"
MODEL_BASE_URL="${EMOTYC_MODEL_BASE_URL:-https://huggingface.co/GwendalTsang/EMOTYC-ONNX/resolve/main}"

MODEL_URL="${EMOTYC_ONNX_URL:-$MODEL_BASE_URL/model.onnx}"
TOKENIZER_URL="${EMOTYC_TOKENIZER_URL:-$MODEL_BASE_URL/tokenizer.json}"
CONFIG_URL="${EMOTYC_CONFIG_URL:-$MODEL_BASE_URL/config.json}"

MODEL_PATH="${EMOTYC_ONNX_PATH:-$MODEL_DIR/model.onnx}"
TOKENIZER_PATH="${EMOTYC_TOKENIZER_PATH:-$MODEL_DIR/tokenizer.json}"
CONFIG_PATH="${EMOTYC_CONFIG_PATH:-$MODEL_DIR/config.json}"

EXPECTED_MODEL_SHA256="${EMOTYC_ONNX_SHA256:-e0c18514933453452929c9f699d68e1fd253414dd44046cc9ea77c445fcfd642}"
EXPECTED_TOKENIZER_SHA256="${EMOTYC_TOKENIZER_SHA256:-18d0f7a36785b8459a9ffa36c0340337ca09450795a721ce0c089168994836ea}"
EXPECTED_CONFIG_SHA256="${EMOTYC_CONFIG_SHA256:-f1e70ec9a0d8f6a1d5209b974efc8e3ee057454fb14b2027ba166d283f22eb22}"

PYTHON_BIN="${PYTHON:-python3}"

usage() {
    cat <<EOF
Usage: bash setup.sh

Installs Python dependencies from requirements.txt, then downloads and verifies
the local EMOTYC ONNX artifacts in model_onnx/.

Options:
  -h, --help     Show this help message.

Environment overrides:
  PYTHON                  Python executable used for pip, default: python3
  EMOTYC_REQUIREMENTS     Requirements file, default: ./requirements.txt
  EMOTYC_MODEL_DIR        Destination directory, default: ./model_onnx
  EMOTYC_MODEL_BASE_URL   Base URL for model.onnx, tokenizer.json, config.json
  EMOTYC_ONNX_URL         ONNX model URL
  EMOTYC_TOKENIZER_URL    tokenizer.json URL
  EMOTYC_CONFIG_URL       config.json URL
  EMOTYC_ONNX_PATH        ONNX destination path
  EMOTYC_TOKENIZER_PATH   tokenizer destination path
  EMOTYC_CONFIG_PATH      config destination path
EOF
}

log() {
    printf '\n==> %s\n' "$*"
}

warn() {
    printf 'WARNING: %s\n' "$*" >&2
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
    usage
    exit 0
fi

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$path" | awk '{print $1}'
    else
        die "sha256sum or shasum is required to verify downloaded artifacts."
    fi
}

validate_file() {
    local path="$1"
    local expected_sha256="$2"
    local digest

    if [[ ! -f "$path" ]]; then
        printf 'file is missing\n'
        return 1
    fi

    digest="$(sha256_file "$path")"
    if [[ -n "$expected_sha256" && "$digest" != "$expected_sha256" ]]; then
        printf 'unexpected sha256: got %s, expected %s\n' "$digest" "$expected_sha256"
        return 1
    fi

    printf 'valid file (sha256=%s)\n' "$digest"
}

install_requirements() {
    [[ -f "$REQUIREMENTS_FILE" ]] || die "Requirements file not found: $REQUIREMENTS_FILE"
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"

    log "Installing Python dependencies from $REQUIREMENTS_FILE"
    "$PYTHON_BIN" -m pip install -r "$REQUIREMENTS_FILE"
}

download_file() {
    local url="$1"
    local tmp_path="$2"

    rm -f "$tmp_path"

    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --show-error --progress-bar --output "$tmp_path" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$tmp_path" "$url"
    else
        die "curl or wget is required to download EMOTYC artifacts."
    fi
}

download_artifact() {
    local name="$1"
    local url="$2"
    local path="$3"
    local expected_sha256="$4"
    local tmp_path="${path}.tmp"
    local validation_message

    mkdir -p "$(dirname "$path")"

    if validation_message="$(validate_file "$path" "$expected_sha256")"; then
        log "$name already present"
        printf 'OK: %s\n%s\n' "$path" "$validation_message"
        return 0
    fi

    if [[ -e "$path" ]]; then
        warn "Local $name is not valid: $validation_message"
        warn "Downloading a fresh copy."
    fi

    log "Downloading $name"
    printf 'URL: %s\n' "$url"
    printf 'Destination: %s\n' "$path"

    download_file "$url" "$tmp_path"

    if ! validation_message="$(validate_file "$tmp_path" "$expected_sha256")"; then
        rm -f "$tmp_path"
        die "Downloaded $name is not valid: $validation_message"
    fi

    mv -f "$tmp_path" "$path"
    printf 'OK: %s\n%s\n' "$path" "$validation_message"
}

cleanup() {
    rm -f "${MODEL_PATH}.tmp" "${TOKENIZER_PATH}.tmp" "${CONFIG_PATH}.tmp"
}
trap cleanup EXIT

install_requirements
download_artifact "ONNX weights" "$MODEL_URL" "$MODEL_PATH" "$EXPECTED_MODEL_SHA256"
download_artifact "tokenizer" "$TOKENIZER_URL" "$TOKENIZER_PATH" "$EXPECTED_TOKENIZER_SHA256"
download_artifact "model config" "$CONFIG_URL" "$CONFIG_PATH" "$EXPECTED_CONFIG_SHA256"
log "Setup complete"
