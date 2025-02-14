get_execution_permissions() {
    local dir="$1"

    # Traverse up until we find an existing directory
    while [[ ! -d "$dir" ]]; do
        dir=$(dirname "$dir") # Move up one level in the directory hierarchy
    done

    # Check write permissions for the closest existing parent directory
    if [[ -w $dir ]]; then
        echo ""
    else
        echo "sudo -E"
        echo "    Elevated permissions are required to create or write to the target directory: $1" >&2
    fi
}
