#!/bin/bash

set -euo pipefail

# =============================================================================
# RL-Swarm Launcher Script - Improved Version
# =============================================================================

# Configuration
readonly ROOT="$PWD"
readonly GENRL_TAG="v0.1.1"
readonly LOG_DIR="$ROOT/logs"
readonly CONFIG_DIR="$ROOT/configs"

# Environment variables with defaults
export IDENTITY_PATH
export GENSYN_RESET_CONFIG
export CONNECT_TO_TESTNET=true
export ORG_ID
export HF_HUB_DOWNLOAD_TIMEOUT=120
export SWARM_CONTRACT="0xFaD7C5e93f28257429569B854151A1B8DCD404c2"
export HUGGINGFACE_ACCESS_TOKEN="None"

# Path configurations
readonly DEFAULT_IDENTITY_PATH="$ROOT/swarm.pem"
IDENTITY_PATH=${IDENTITY_PATH:-$DEFAULT_IDENTITY_PATH}
DOCKER=${DOCKER:-""}
GENSYN_RESET_CONFIG=${GENSYN_RESET_CONFIG:-""}
CPU_ONLY=${CPU_ONLY:-""}
ORG_ID=${ORG_ID:-""}

# Color codes
readonly GREEN_TEXT="\033[32m"
readonly BLUE_TEXT="\033[34m"
readonly RED_TEXT="\033[31m"
readonly YELLOW_TEXT="\033[33m"
readonly RESET_TEXT="\033[0m"

# =============================================================================
# Utility Functions
# =============================================================================

# Logging functions
log_info() {
    echo -e "${GREEN_TEXT}[INFO]${RESET_TEXT} $1"
}

log_warn() {
    echo -e "${YELLOW_TEXT}[WARN]${RESET_TEXT} $1"
}

log_error() {
    echo -e "${RED_TEXT}[ERROR]${RESET_TEXT} $1"
}

log_debug() {
    echo -e "${BLUE_TEXT}[DEBUG]${RESET_TEXT} $1"
}

# Initialize directories
init_directories() {
    log_info "Initializing directories..."
    
    local directories=("$LOG_DIR" "$CONFIG_DIR")
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
}

# Docker volume setup
setup_docker_volumes() {
    if [[ -n "$DOCKER" ]]; then
        log_info "Setting up Docker volumes..."
        
        local volumes=(
            "/home/gensyn/rl_swarm/modal-login/temp-data"
            "/home/gensyn/rl_swarm/keys"
            "/home/gensyn/rl_swarm/configs"
            "/home/gensyn/rl_swarm/logs"
        )
        
        for volume in "${volumes[@]}"; do
            if [[ -d "$volume" ]]; then
                sudo chown -R 1001:1001 "$volume"
                log_info "Set ownership for volume: $volume"
            fi
        done
    fi
}

# =============================================================================
# Node.js and Yarn Installation
# =============================================================================

install_nodejs() {
    if ! command -v node > /dev/null 2>&1; then
        log_info "Node.js not found. Installing NVM and latest Node.js..."
        
        export NVM_DIR="$HOME/.nvm"
        if [[ ! -d "$NVM_DIR" ]]; then
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
        fi
        
        # Source NVM
        [[ -s "$NVM_DIR/nvm.sh" ]] && \. "$NVM_DIR/nvm.sh"
        [[ -s "$NVM_DIR/bash_completion" ]] && \. "$NVM_DIR/bash_completion"
        
        nvm install node
        log_info "Node.js installed successfully"
    else
        log_info "Node.js is already installed: $(node -v)"
    fi
}

install_yarn() {
    if ! command -v yarn > /dev/null 2>&1; then
        log_info "Installing Yarn..."
        
        # Detect OS and install accordingly
        if grep -qi "ubuntu" /etc/os-release 2> /dev/null || uname -r | grep -qi "microsoft"; then
            log_info "Detected Ubuntu/WSL. Installing Yarn via apt..."
            curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
            echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
            sudo apt update && sudo apt install -y yarn
        else
            log_info "Installing Yarn globally with npm..."
            npm install -g --silent yarn
        fi
        
        log_info "Yarn installed successfully"
    else
        log_info "Yarn is already installed: $(yarn --version)"
    fi
}

# =============================================================================
# Modal Login Server Setup
# =============================================================================

setup_modal_login() {
    log_info "Setting up modal login server..."
    
    cd modal-login || {
        log_error "Failed to change to modal-login directory"
        return 1
    }
    
    # Update environment file
    local env_file="$ROOT/modal-login/.env"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "3s/.*/SMART_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$env_file"
    else
        sed -i "3s/.*/SMART_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$env_file"
    fi
    
    # Install dependencies and build (skip if Docker)
    if [[ -z "$DOCKER" ]]; then
        log_info "Installing dependencies and building server..."
        yarn install --immutable
        yarn build > "$LOG_DIR/yarn.log" 2>&1
    fi
    
    # Start server
    log_info "Starting modal login server..."
    yarn start >> "$LOG_DIR/yarn.log" 2>&1 &
    
    local server_pid=$!
    log_info "Started server process: $server_pid"
    sleep 5
    
    # Open browser
    if [[ -z "$DOCKER" ]]; then
        if open http://localhost:3000 2> /dev/null; then
            log_info "Successfully opened http://localhost:3000 in your default browser"
        else
            log_warn "Failed to open http://localhost:3000. Please open it manually"
        fi
    else
        log_info "Please open http://localhost:3000 in your host browser"
    fi
    
    cd ..
}

# Wait for user data
wait_for_user_data() {
    log_info "Waiting for modal userData.json to be created..."
    
    local user_data_file="modal-login/temp-data/userData.json"
    while [[ ! -f "$user_data_file" ]]; do
        sleep 5
    done
    
    log_info "Found userData.json. Proceeding..."
    
    # Extract ORG_ID
    ORG_ID=$(awk 'BEGIN { FS = "\"" } !/^[ \t]*[{}]/ { print $(NF - 1); exit }' "$user_data_file")
    log_info "Your ORG_ID is set to: $ORG_ID"
    export ORG_ID
}

# Wait for API key activation
wait_for_api_activation() {
    log_info "Waiting for API key to become activated..."
    
    while true; do
        local status
        status=$(curl -s "http://localhost:3000/api/get-api-key-status?orgId=$ORG_ID" 2>/dev/null || echo "error")
        
        if [[ "$status" == "activated" ]]; then
            log_info "API key is activated! Proceeding..."
            break
        else
            log_info "Waiting for API key to be activated..."
            sleep 5
        fi
    done
}

# =============================================================================
# Python Dependencies Installation
# =============================================================================

install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install required packages
    local packages=(
        "gensyn-genrl==0.1.4"
        "reasoning-gym>=0.1.20"
        "trl"
        "hivemind@git+https://github.com/learning-at-home/hivemind@4d5c41495be082490ea44cce4e9dd58f9926bb4e"
    )
    
    for package in "${packages[@]}"; do
        log_info "Installing $package..."
        pip install "$package"
    done
    
    log_info "Python dependencies installed successfully"
}

# Patch hivemind p2p daemon timeout
patch_hivemind_timeout() {
    log_info "Checking for hivemind p2p daemon patch..."
    
    # Common paths where the file might be located
    local possible_paths=(
        "$ROOT/.venv/lib/python3.12/site-packages/hivemind/p2p/p2p_daemon.py"
        "$HOME/rl-swarm/.venv/lib/python3.12/site-packages/hivemind/p2p/p2p_daemon.py"
        "$(python -c 'import hivemind; print(hivemind.__file__.replace("__init__.py", "p2p/p2p_daemon.py"))' 2>/dev/null || echo '')"
    )
    
    # Try to find the file
    local daemon_file=""
    for path in "${possible_paths[@]}"; do
        if [[ -f "$path" ]]; then
            daemon_file="$path"
            break
        fi
    done
    
    if [[ -n "$daemon_file" ]]; then
        log_info "Found hivemind p2p daemon file: $daemon_file"
        
        # Check if patch is already applied
        if grep -q "startup_timeout: float = 120" "$daemon_file"; then
            log_info "Timeout patch already applied"
        else
            log_info "Applying timeout patch (15s -> 120s)..."
            # Create backup first
            cp "$daemon_file" "$daemon_file.backup"
            
            # Apply patch
            sed -i 's/startup_timeout: float = 15/startup_timeout: float = 120/' "$daemon_file"
            
            # Verify patch was applied
            if grep -q "startup_timeout: float = 120" "$daemon_file"; then
                log_info "Timeout patch applied successfully"
            else
                log_warn "Failed to apply timeout patch"
                # Restore backup if patch failed
                mv "$daemon_file.backup" "$daemon_file"
            fi
        fi
    else
        log_warn "Hivemind p2p daemon file not found. Patch will be skipped."
        log_debug "Searched paths: ${possible_paths[*]}"
    fi
}

# =============================================================================
# Configuration Management
# =============================================================================

setup_config() {
    log_info "Setting up configuration..."
    
    local config_file="$CONFIG_DIR/rg-swarm.yaml"
    local default_config="$ROOT/rgym_exp/config/rg-swarm.yaml"
    
    if [[ -f "$config_file" ]]; then
        if ! cmp -s "$default_config" "$config_file"; then
            if [[ -z "$GENSYN_RESET_CONFIG" ]]; then
                log_warn "Found differences in rg-swarm.yaml. Set GENSYN_RESET_CONFIG to reset to default."
            else
                log_info "Backing up existing config and resetting to default..."
                mv "$config_file" "$config_file.bak"
                cp "$default_config" "$config_file"
            fi
        fi
    else
        log_info "Copying default configuration..."
        cp "$default_config" "$config_file"
    fi
    
    # Set permissions for Docker
    if [[ -n "$DOCKER" ]]; then
        sudo chmod -R 0777 "$CONFIG_DIR"
    fi
}

# =============================================================================
# Error Handling and Cleanup
# =============================================================================

cleanup() {
    log_info "Shutting down trainer..."
    exit 0
}

# =============================================================================
# Main Execution
# =============================================================================

display_banner() {
    echo -e "\033[38;5;224m"
    cat << "EOF"
    ██████  ██            ███████ ██     ██  █████  ██████  ███    ███
    ██   ██ ██            ██      ██     ██ ██   ██ ██   ██ ████  ████
    ██████  ██      █████ ███████ ██  █  ██ ███████ ██████  ██ ████ ██
    ██   ██ ██                 ██ ██ ███ ██ ██   ██ ██   ██ ██  ██  ██
    ██   ██ ███████       ███████  ███ ███  ██   ██ ██   ██ ██      ██

    From Gensyn - Improved Version

EOF
    echo -e "$RESET_TEXT"
}

main() {
    # Set up trap for cleanup
    trap cleanup EXIT
    
    # Initialize
    display_banner
    init_directories
    setup_docker_volumes
    
    # Testnet connection setup
    if [[ "$CONNECT_TO_TESTNET" == "true" ]]; then
        log_info "Setting up testnet connection..."
        
        install_nodejs
        install_yarn
        setup_modal_login
        wait_for_user_data
        wait_for_api_activation
    fi
    
    # Install dependencies
    install_python_deps
    patch_hivemind_timeout
    setup_config
    
    # Final setup
    log_info "Configuration complete!"
    log_info "Good luck in the swarm!"
    log_debug "Remember to star the repo on GitHub! --> https://github.com/gensyn-ai/rl-swarm"
    
    # Launch the swarm
    log_info "Launching RL-Swarm..."
    python -m rgym_exp.runner.swarm_launcher \
        --config-path "$ROOT/rgym_exp/config" \
        --config-name "rg-swarm.yaml"
    
    # Keep script running
    wait
}

# Execute main function
main "$@"
