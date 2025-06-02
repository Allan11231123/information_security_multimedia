#!/usr/bin/env bash
# Generate a random string of specified length

set -euo pipefail
# set -x
OUTPUT_DIR=""

# echo "Passed arguments:"
# printf '<%s> ' "$@"

if [[ "$1" == "-o" ]]; then
  if [ $# -lt 3 ]; then
    echo "Usage: $0 [-o output_dir] <length>" >&2
    exit 1
  fi
  OUTPUT_DIR="$2"
  shift 2
elif [[ "${1:-}" == --output-dir=* ]]; then
  OUTPUT_DIR="${1#--output-dir=}"
  shift 1
elif [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: $0 [-o output_dir] <length>"
  echo "Generate a random string of specified length and save it to a file."
  echo "Options:"
  echo "  -o, --output-dir <output_dir>   Specify the directory to save the output file."
  exit 0
fi

if [ $# -eq 0 ]; then
  echo "Usage: $0 [-o output_dir] <length>" >&2
  exit 1
fi

if [ -n "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
parse_size_to_bytes() {
  local arg="$1"
  if [[ "$arg" =~ ^([0-9]+)([bBkKmMgG][bB]?)$ ]]; then
    local num="${BASH_REMATCH[1]}"
    local unit="${BASH_REMATCH[2]^^}"
    local multiplier=1

    case "$unit" in
      B)
        multiplier=1
        ;;
      KB)
        multiplier=$((1024))
        ;;
      MB)
        multiplier=$((1024 * 1024))
        ;;
      GB)
        multiplier=$((1024 * 1024 * 1024))
        ;;
      *)
        echo "error: unsupported unit “$unit”, please use B, KB, MB or GB" >&2
        exit 1
        ;;
    esac

    echo $(( num * multiplier ))
  else
    echo "error: argument “$arg” has improper format, correct: <integer><unit> (ex: 32B, 4KB, 1MB)" >&2
    exit 1
  fi
}

for raw_size in "$@"; do
  echo "Processing size: $raw_size"
  size=$(parse_size_to_bytes "$raw_size")
  if [ -z "$size" ]; then
    echo "error: failed to parse size from $raw_size" >&2
    exit 1
  fi

  if [ -n "$OUTPUT_DIR" ]; then
    output_file="$OUTPUT_DIR/rand_str_${size}.txt"
  else
    output_file="rand_str_${size}.txt"
  fi
  set +o pipefail
  # Generate random string and save to file
  tr -dc 'a-zA-Z0-9' < /dev/urandom | head -c "$size" > "$output_file"
  set -o pipefail

  echo "Generated random string of length $size and saved to $output_file"
done