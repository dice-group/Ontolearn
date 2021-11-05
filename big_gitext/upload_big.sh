#!/usr/bin/env bash

# This script is used to upload resources to an external server
# Requires curl

set -eu

FTP=ftp://hobbitdata.informatik.uni-leipzig.de/public/GitExt/OntoPy
HTTP=https://hobbitdata.informatik.uni-leipzig.de/GitExt/OntoPy

abs2rel() { perl -l -MFile::Spec -e'print File::Spec->abs2rel(@ARGV)' "$@"; }

git_dir="$(git rev-parse --git-dir)"
repo_root="$(git rev-parse --show-toplevel)"
repo_root="$(abs2rel "$repo_root")"
if [[ -z "$git_dir" ]] || [[ -z "$repo_root" ]]; then
   exit 2
fi

find_content_length() {
    echo "$1"|awk -v IGNORECASE=1 '{ sub("\r$", "") } $1 == "Content-Length:" { print $2; exit }'
}

if ! command -v sha256sum >/dev/null; then
    sha256sum() {
        sha256="$(openssl sha256 "$1")"
        echo "$(echo "$sha256"|awk -v FS='= ' '{print $NF}')""  ""$1"
    }
fi


upload_file() {
    if [[ -f "$1" ]]; then
        :
    else
        echo "Error: \`$1' is not a regular file"
        exit 1
    fi

    if [[ "$1" == *.link ]]; then
        echo "Error: \`$1' is a .link file"
        exit 1
    fi

    oid="$(git hash-object "$1")"
    if [[ -z "$oid" ]]; then
        echo "Error: git hash-object failed"
        exit 2
    fi

    sha256="$(sha256sum "$1" | awk '{print $1}')"
    if [[ -z "$sha256" ]]; then
        echo "Error: sha256sum failed"
        exit 2
    fi

    size="$( ( stat --printf="%s" "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null ) | awk '{print $1}')"
    if [[ -z "$size" ]]; then
        echo "Error: stat failed"
        exit 2
    fi


    linkfile="$1.link"
    : >"$linkfile"
    git add -N "$linkfile"
    filepath="$(git ls-files --full-name "$linkfile")"
    if [[ -z "$filepath" ]]; then
        exit 2
    fi

    filepath="${filepath%.link}"
    basename="${filepath##*/}"
    dirname="${filepath%"$basename"}"
    root="${basename%.*}"
    if [[ -z "$root" ]]; then
        root="$basename"
        ext=""
    else
        ext="${basename#"$root"}"
    fi


    exec 6>&1 >"$linkfile"

    echo "#% GitExt 0.1"
    echo "path:$filepath"
    echo "oid:$oid"
    echo "sha256sum:$sha256"
    echo "size:$size"

    exec 1>&6 6>&-

    bfpath="$git_dir/big_files/$(echo "$oid"|cut -b1-2)/$(echo "$oid"|cut -b3-4)/"
    mkdir -p "$bfpath"
    newf="$bfpath$oid"
    echo "Uploading $filepath..."
    cp "$1" "$newf"
    ret=0
    check_size=-1
    upload_test="$(curl -f -s -I "$HTTP/$dirname$root/$oid$ext")" || ret=$?
    if [[ "$ret" -eq 0 ]]; then
        check_size="$(find_content_length "$upload_test")"
    fi
    if [[ "$check_size" -ne "$size" ]]; then
        curl -f -n -T "$1" -C- --ssl --ftp-create-dirs "$FTP/$dirname$root/$oid$ext"
    else
        echo '(cached)'
    fi
    upload_test="$(curl -f -s -I "$HTTP/$dirname$root/$oid$ext")" || ret=$?
    if [[ "$ret" -eq 0 ]]; then
        check_size="$(find_content_length "$upload_test")"
    fi
    if [[ -z "$check_size" ]] || [[ "$check_size" -ne "$size" ]]; then
        echo "Upload failed, size mismatch"
        exit 2
    fi

    git add "$linkfile"
    if ! grep -qFx "$filepath" "$repo_root/.gitignore"; then
        echo "$filepath" >>"$repo_root/.gitignore"
        git add "$repo_root/.gitignore"
    fi
}

if [[ "$#" -eq 0 ]]; then
    echo "syntax: ./big_gitext/upload_big.sh -A|<filename...>"
    exit 1
fi

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    cat <<EOT
Usage:
  ./big_gitext/upload_big.sh -A|<filename...>

Description:
  Upload files to an external server that should not be stored in Git
  directly. Creates .link files that contain metadata about the external
  file.

  Requires curl.

Arguments:
  -A            Upload all files for which a .link file already exists
  filename(s)   Upload these files and create .link files

Example:
  ./big_gitext/upload_big.sh model.pt

EOT
    exit
fi

if [[ "$1" == "-A" ]] || [[ "$1" == "--all" ]]; then
    shopt -s globstar nullglob dotglob
    files=()
    for lf in "$repo_root/"**/*.link; do
        file="${lf%.link}"
        if [[ -e "$file" ]]; then
            files+=("$file")
        else
            echo "Link without file: ${lf#./}"
        fi
    done
else
    files=("$@")
fi

for file in "${files[@]}"; do
    upload_file "$file"
done
echo "Done."
