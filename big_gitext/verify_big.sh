#!/usr/bin/env bash

# This script is used to quickly verify resources on an external server
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

find_content_length() {
    echo "$1"|awk -v IGNORECASE=1 '{ sub("\r$", "") } $1 == "Content-Length:" { print $2; exit }'
}


verify_file() {
    if [[ "$1" == *.link ]]; then
        f="$1"
    else
        f="$1".link
    fi

    if [[ -f "$f" ]]; then
        :
    else
        echo "Error: \`$f' is not a regular file"
        return 1
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
        return 2
    fi

    oid="${link_info["oid"]}"
    if [[ -z "$oid" ]]; then
        echo "Error: oid missing"
        return 2
    fi

    filepath="${link_info["path"]}"
    basename="${filepath##*/}"
    dirname="${filepath%"$basename"}"
    root="${basename%.*}"
    if [[ -z "$root" ]]; then
        root="$basename"
        ext=""
    else
        ext="${basename#"$root"}"
    fi

    size="${link_info["size"]}"

    echo "Verifying $filepath..."
    upload_test="$(curl -f -s -I "$HTTP/$dirname$root/$oid$ext")" || ret=$?
    if [[ "$ret" -ne 0 ]]; then
        echo "Verify failed, file not found"
        return 2
    else
        check_size="$(find_content_length "$upload_test")"
        if [[ -z "$check_size" ]] || [[ "$check_size" -ne "$size" ]]; then
            echo "Verify failed, size mismatch"
            return 2
        fi
    fi

}

if [[ "$#" -eq 0 ]]; then
    echo "syntax: ./big_gitext/verify_big.sh -A|<filename...>"
    exit 1
fi

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    cat <<EOT
Usage:
  ./big_gitext/verify_big.sh -A|<filename...>

Description:
  Quickly verify if the given files are stored on an external server
  by checking if the oid file given in the .link file has the expected
  size. Attention: Does not verify the checksum.

  Requires curl.

Arguments:
  -A            Verify all .link files
  filename(s)   Verify the specified .link files only


Example:
  ./big_gitext/verify_big.sh model.pt

EOT
    exit
fi

if [[ "$1" == "-A" ]] || [[ "$1" == "--all" ]]; then
    shopt -s globstar nullglob dotglob
    files=("$repo_root/"**/*.link)
else
    files=("$@")
fi

# check if the server is reachable
echo -n "Testing connection to $HTTP ..."
curl -f -s -S -I -o /dev/null "$HTTP" && echo "ok" || exit $?

ret=0
for file in "${files[@]}"; do
    verify_file "$file" || ret=$(( ret + $? ))
done
if [[ "$ret" -gt 0 ]]; then
    echo "Errors."
    exit "$ret"
fi

echo "Done."
