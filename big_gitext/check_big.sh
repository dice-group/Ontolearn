#!/usr/bin/env bash

# This script is used to check resources that should be stored outside
# of Git on an external server

set -eu

abs2rel() { perl -l -MFile::Spec -e'print File::Spec->abs2rel(@ARGV)' "$@"; }

git_dir="$(git rev-parse --git-dir)"
repo_root="$(git rev-parse --show-toplevel)"
repo_root="$(abs2rel "$repo_root")"
if [[ -z "$git_dir" ]] || [[ -z "$repo_root" ]]; then
    exit 2
fi

if ! command -v sha256sum >/dev/null; then
    sha256sum() {
        sha256="$(openssl sha256 "$1")"
        echo "$(echo "$sha256"|awk -v FS='= ' '{print $NF}')""  ""$1"
    }
fi


check_file() {
    :
    if [[ "$1" == *.link ]]; then
        f="$1"
    else
        f="$1".link
    fi
    o="${f%.link}"

    declare -A link_info
    declare -A local_info

    noid=
    if [[ -f "$o" ]]; then
        noid="$(git hash-object "$o")"
        if [[ -z "$noid" ]]; then
            echo "Error: git hash-object failed"
            exit 2
        fi
        local_info["oid"]="$noid"

        sha256="$(sha256sum "$o" | awk '{print $1}')"
        if [[ -z "$sha256" ]]; then
            echo "Error: sha256sum failed"
            exit 2
        fi
        local_info["sha256sum"]="$sha256"

        size="$( ( stat --printf="%s" "$o" 2>/dev/null || stat -f%z "$o" 2>/dev/null ) | awk '{print $1}')"
        if [[ -z "$size" ]]; then
            echo "Error: stat failed"
            exit 2
        fi
        local_info["size"]="$size"

        nbfpath="$git_dir/big_files/$(echo "$noid"|cut -b1-2)/$(echo "$noid"|cut -b3-4)/"
        nlocalf="$nbfpath$noid"
        if [[ -f "$nlocalf" ]]; then
            local_info["cached"]=1
        else
            local_info["cached"]=0
        fi
    fi

    if [[ "${local_info["cached"]-0}" -eq 1 ]]; then
        _c=C
        _m="$_c"
        _ct=", but is cached locally"
        _mt=", and has a local copy"
    else
        _c=" "
        _m=M
        _ct=""
        _mt=", and is not checked in"
    fi

    if [[ -f "$f" ]]; then
        while IFS=: read -r k v; do
            if [[ -n "$k" ]]; then
                link_info["$k"]="$v"
            fi
        done <"$f"
    else
        if [[ -f "$o" ]]; then
            if [[ "$short_mode" -eq 1 ]]; then
                echo "?$_c ${o#./}"
            else
                echo "  missing .link: \`${o#./}'$_ct"
            fi
        else
            if [[ "$short_mode" -eq 1 ]]; then
                echo "!! ${o#./}"
            else
                echo "Error: \`${o#./}' does not exist"
            fi
        fi
        return
    fi


    hdr="#% GitExt 0.1"
    if [[ "${link_info["$hdr"]-x}" != "" ]]; then
        if [[ "$short_mode" -eq 1 ]]; then
            echo "!$_c ${o#./}"
        else
            echo "Error: \`${o#./}' has no valid .link file$_ct"
        fi
        return
    fi

    oid="${link_info["oid"]}"
    if [[ -z "$oid" ]]; then
        if [[ "$short_mode" -eq 1 ]]; then
            echo "!$_c ${o#./}"
        else
            echo "Error: \`${f#./}': oid missing$_ct"
        fi
        return
    fi


    bfpath="$git_dir/big_files/$(echo "$oid"|cut -b1-2)/$(echo "$oid"|cut -b3-4)/"
    localf="$bfpath$oid"
    if [[ -f "$localf" ]]; then
        link_info["cached"]=1
        _d=U
        __dt=" is cached locally"
        _dt=", but$__dt"
        _dta=", and$__dt"
    else
        link_info["cached"]=0
        _d=D
        _dt=""
        _dta=""
    fi

    if [[ -f "$o" ]]; then
        if [[ "$sha256" == "${link_info["sha256sum"]}" ]]; then
            if [[ "$auto_mode" -eq 1 ]]; then
                :
            else
                if [[ "$short_mode" -eq 1 ]]; then
                    echo ".$_c ${o#./}"
                else
                    echo "  unchanged from .link: \`${o#./}'$_dta"
                fi
            fi
        else
            # files differ
            :
            if [[ "$short_mode" -eq 1 ]]; then
                echo "M$_m ${o#./}"
            else
                echo "  file differs: \`${o#./}'$_mt"
            fi
        fi
    else
        if [[ "$short_mode" -eq 1 ]]; then
            echo "$_d  ${o#./}"
        else
            echo "  not downloaded: \`${o#./}'$_dt"
        fi
    fi
}

short_mode=0
if [[ "$#" -ge 1 ]]; then
    if [[ "$1" == "-s" ]] || [[ "$1" == "--short" ]]; then
        short_mode=1
        shift
    fi
fi

auto_mode=0
if [[ "$#" -eq 0 ]]; then
    set -- -A
    auto_mode=1
fi

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    cat <<EOT
Usage:
  ./big_gitext/check_big.sh [-s] -A|<filename...>

Description:
  Check files that should not be stored in Git directly. Uses .link
  files that contain metadata about the external file to check if
  they are changed.

Options:
  -s            Show short summary.

Arguments:
  -A            Check all files for which a .link file exists
  filename(s)   Check the specified files only

Short summary legend:
  D             not downloaded
  U             not checked out
  .             unchanged from .link file
  C             local copy exists
  M             differs from .link file

Example:
  ./big_gitext/check_big.sh -s
  ./big_gitext/check_big.sh model.pt

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
    check_file "$file"
done

if [[ "$short_mode" -eq 0 ]]; then
    echo "Done."
fi
