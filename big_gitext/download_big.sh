#!/usr/bin/env bash

# This script is used to download resources from an external server
# Requires curl

set -eu

HTTP=https://hobbitdata.informatik.uni-leipzig.de/GitExt/OntoPy

abs2rel() { perl -l -MFile::Spec -e'print File::Spec->abs2rel(@ARGV)' "$@"; }

git_dir="$(git rev-parse --git-dir 2>/dev/null || :)"
repo_root="$(git rev-parse --show-toplevel 2>/dev/null || :)"
repo_root="$(abs2rel "$repo_root")"
if [[ -z "$git_dir" ]] || [[ -z "$repo_root" ]]; then
    echo "No git detected"
    repo_root=.
fi

if ! command -v sha256sum >/dev/null; then
    sha256sum() {
        sha256="$(openssl sha256 "$1")"
        echo "$(echo "$sha256"|awk -v FS='= ' '{print $NF}')""  ""$1"
    }
fi


download_file() {
    :
    if [[ "$1" == *.link ]]; then
        f="$1"
    else
        f="$1".link
    fi
    o="${f%.link}"

    if [[ -f "$f" ]]; then
        :
    else
        echo "Error: \`$f' is not a regular file"
        exit 1
    fi

    declare -A link_info
    while IFS=: read -r k v; do
        if [[ -n "$k" ]]; then
            link_info["$k"]="$v"
        fi
    done <"$f"

    hdr="#% GitExt 0.1"
    if [[ "${link_info["$hdr"]-x}" != "" ]]; then
        echo "Error: no valid .link file"
        exit 2
    fi

    oid="${link_info["oid"]}"
    if [[ -z "$oid" ]]; then
        echo "Error: oid missing"
        exit 2
    fi


    if [[ -n "$git_dir" ]]; then
        bfpath="$git_dir/big_files/$(echo "$oid"|cut -b1-2)/$(echo "$oid"|cut -b3-4)/"
        localf="$bfpath$oid"
    else
        localf=
    fi

    filepath="${link_info["path"]}"

    skip=0
    if [[ -f "$o" ]]; then
        sha256="$(sha256sum "$o" | awk '{print $1}')"
        if [[ "$sha256" == "${link_info["sha256sum"]}" ]]; then
            echo "Already exists: $filepath..."
            skip=1
        else
            mv "$o" "$o"~~
        fi
    fi

    backup=0
    if [[ -f "$o"~~ ]]; then
        backup=1
    fi

    basename="${filepath##*/}"
    dirname="${filepath%"$basename"}"
    root="${basename%.*}"
    if [[ -z "$root" ]]; then
        root="$basename"
        ext=""
    else
        ext="${basename#"$root"}"
    fi

    if [[ "$skip" -eq 0 ]]; then
        echo "Downloading $filepath..."
        if [[ -n "$localf" ]] && [[ -f "$localf" ]]; then
            cp "$localf" "$o"
            echo '(cached)'
        else
            curl -f -o "$o".part -C- "$HTTP/$dirname$root/$oid$ext"
            mv "$o".part "$o"
        fi

        sha256="$(sha256sum "$o" | awk '{print $1}')"
        if [[ "$sha256" != "${link_info["sha256sum"]}" ]]; then
            echo "Error: sha256sum failed"
            exit 2
        fi
    fi

    size="$( ( stat --printf="%s" "$o" 2>/dev/null || stat -f%z "$o" 2>/dev/null ) | awk '{print $1}')"
    if [[ "$size" -ne "${link_info["size"]}" ]]; then
        echo "Error: size mismatch"
        exit 2
    fi

    if [[ -n "$localf" ]] && [[ ! -f "$localf" ]]; then
        mkdir -p "$bfpath"
        cp "$o" "$localf"
    fi

    if [[ "$backup" -eq 1 ]]; then
        if cmp -s "o" "$o"~~; then
            rm "$o"~~
        else
            mv "$o"~~ "$o"~
        fi
    fi
}

if [[ "$#" -eq 0 ]]; then
    echo "syntax: ./big_gitext/download_big.sh -A|<filename...>"
    exit 1
fi

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    cat <<EOT
Usage:
  ./big_gitext/download_big.sh -A|<filename...>

Description:
  Download files from an external server that were not stored in Git
  directly. Uses .link files that contain metadata about the external
  file to find the download information.

  Requires curl.

Arguments:
  -A            Download all files for which a .link file exists
  filename(s)   Download the specified files only

Example:
  ./big_gitext/download_big.sh model.pt

EOT
    exit
fi

if [[ "$1" == "-A" ]] || [[ "$1" == "--all" ]]; then
    shopt -s globstar nullglob dotglob
    files=("$repo_root/"**/*.link)
else
    files=("$@")
fi

for file in "${files[@]}"; do
    download_file "$file"
done
echo "Done."
