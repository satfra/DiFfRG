expandPath() {
    local path
    local -a pathElements resultPathElements
    IFS=':' read -r -a pathElements <<<"$1"
    : "${pathElements[@]}"
    for path in "${pathElements[@]}"; do
        : "$path"
        case $path in
        "~+"/*)
            path=$PWD/${path#"~+/"}
            ;;
        "~-"/*)
            path=$OLDPWD/${path#"~-/"}
            ;;
        "~"/*)
            path=$HOME/${path#"~/"}
            ;;
        "~"*)
            username=${path%%/*}
            username=${username#"~"}
            IFS=: read -r _ _ _ _ _ homedir _ < <(getent passwd "$username")
            if [[ $path = */* ]]; then
                path=${homedir}/${path#*/}
            else
                path=$homedir
            fi
            ;;
        esac
        resultPathElements+=("$path")
    done
    local result
    printf -v result '%s:' "${resultPathElements[@]}"
    printf '%s\n' "${result%:}"
}
