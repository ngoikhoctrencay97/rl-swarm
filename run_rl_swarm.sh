#!/bin/bash

ROOT=$PWD

RED='\033[0;31m'
GREEN='\033[0;32m'
PURPLE='\033[0;95m'
BLUE='\033[0;94m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

export PUB_MULTI_ADDRS
export PEER_MULTI_ADDRS
export HOST_MULTI_ADDRS
export IDENTITY_PATH
export ORG_ID
export HF_HUB_DOWNLOAD_TIMEOUT=120
export TUNNEL_TYPE=""

DEFAULT_PUB_MULTI_ADDRS=""
PUB_MULTI_ADDRS=${PUB_MULTI_ADDRS:-$DEFAULT_PUB_MULTI_ADDRS}

DEFAULT_PEER_MULTI_ADDRS="/ip4/38.101.215.13/tcp/30002/p2p/QmQ2gEXoPJg6iMBSUFWGzAabS2VhnzuS782Y637hGjfsRJ"
PEER_MULTI_ADDRS=${PEER_MULTI_ADDRS:-$DEFAULT_PEER_MULTI_ADDRS}

DEFAULT_HOST_MULTI_ADDRS="/ip4/0.0.0.0/tcp/38331"
HOST_MULTI_ADDRS=${HOST_MULTI_ADDRS:-$DEFAULT_HOST_MULTI_ADDRS}

DEFAULT_IDENTITY_PATH="$ROOT"/swarm.pem
IDENTITY_PATH=${IDENTITY_PATH:-$DEFAULT_IDENTITY_PATH}

SMALL_SWARM_CONTRACT="0x69C6e1D608ec64885E7b185d39b04B491a71768C"
BIG_SWARM_CONTRACT="0x6947c6E196a48B77eFa9331EC1E3e45f3Ee5Fd58"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  if command -v apt &>/dev/null; then
    echo -e "${CYAN}${BOLD}[✓] Debian/Ubuntu detected. Installing build-essential, gcc, g++...${NC}"
    sudo apt update > /dev/null 2>&1
    sudo apt install -y build-essential gcc g++ > /dev/null 2>&1

  elif command -v yum &>/dev/null; then
    echo -e "${CYAN}${BOLD}[✓] RHEL/CentOS detected. Installing Development Tools...${NC}"
    sudo yum groupinstall -y "Development Tools" > /dev/null 2>&1
    sudo yum install -y gcc gcc-c++ > /dev/null 2>&1

  elif command -v pacman &>/dev/null; then
    echo -e "${CYAN}${BOLD}[✓] Arch Linux detected. Installing base-devel...${NC}"
    sudo pacman -Sy --noconfirm base-devel gcc > /dev/null 2>&1

  else
    echo -e "${RED}${BOLD}[✗] Linux detected but unsupported package manager.${NC}"
    exit 1
  fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo -e "${CYAN}${BOLD}[✓] macOS detected. Installing Xcode Command Line Tools...${NC}"
  xcode-select --install > /dev/null 2>&1

else
  echo -e "${RED}${BOLD}[✗] Unsupported OS: $OSTYPE${NC}"
  exit 1
fi

if command -v gcc &>/dev/null; then
  export CC=$(command -v gcc)
  echo -e "${CYAN}${BOLD}[✓] Exported CC=$CC${NC}"
else
  echo -e "${RED}${BOLD}[✗] gcc not found. Please install it manually.${NC}"
fi

[ -f cuda.sh ] && rm cuda.sh; curl -o cuda.sh https://raw.githubusercontent.com/zunxbt/gensyn-testnet/main/cuda.sh && chmod +x cuda.sh && . ./cuda.sh

# Thiết lập swarm mặc định là A (Math)
echo -e "\n${CYAN}${BOLD}[✓] Selecting swarm: Math (default)${NC}"
USE_BIG_SWARM=false
SWARM_CONTRACT="$SMALL_SWARM_CONTRACT"

# Thiết lập số lượng tham số mặc định là 32 tỷ
echo -e "\n${CYAN}${BOLD}[✓] Selecting parameter count: 32 billion (default)${NC}"
PARAM_B=32

cleanup() {
    echo -e "${YELLOW}${BOLD}[✓] Shutting down processes...${NC}"
    kill $SERVER_PID 2>/dev/null || true
    kill $TUNNEL_PID 2>/dev/null || true
    exit 0
}

trap cleanup INT

sleep 2

if [ -f "modal-login/temp-data/userData.json" ]; then
    cd modal-login

    echo -e "\n${CYAN}${BOLD}[✓] Installing dependencies with npm. This may take a few minutes, depending on your internet speed...${NC}"
    npm install --legacy-peer-deps
    
    echo -e "\n${CYAN}${BOLD}[✓] Starting the development server...${NC}"
    if ! command -v ss &>/dev/null; then
      echo -e "${YELLOW}[!] 'ss' not found. Attempting to install 'iproute2'...${NC}"
      if command -v apt &>/dev/null; then
        sudo apt update && sudo apt install -y iproute2
      elif command -v yum &>/dev/null; then
        sudo yum install -y iproute
      elif command -v pacman &>/dev/null; then
        sudo pacman -Sy iproute2
      else
        echo -e "${RED}[✗] Could not install 'ss'. Package manager not found.${NC}"
        exit 1
      fi
    fi
    
    PORT_LINE=$(ss -ltnp | grep ":3000 ")
    if [ -n "$PORT_LINE" ]; then
      PID=$(echo "$PORT_LINE" | grep -oP 'pid=\K[0-9]+')
      if [ -n "$PID" ]; then
        echo -e "${YELLOW}[!] Port 3000 is in use. Killing process: $PID${NC}"
        kill -9 $PID
        sleep 2
      fi
    fi
    
    npm run dev > server.log 2>&1 &
    SERVER_PID=$!
    MAX_WAIT=30  
    
    for ((i = 0; i < MAX_WAIT; i++)); do
        if grep -q "Local:        http://localhost:" server.log; then
            PORT=$(grep "Local:        http://localhost:" server.log | sed -n 's/.*http:\/\/localhost:\([0-9]*\).*/\1/p')
            if [ -n "$PORT" ]; then
                echo -e "${GREEN}${BOLD}[✓] Server is running successfully on port $PORT.${NC}"
                break
            fi
        fi
        sleep 1
    done
    
    if [ $i -eq $MAX_WAIT ]; then
        echo -e "${RED}${BOLD}[✗] Timeout waiting for server to start.${NC}"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    
    cd ..

    ORG_ID=$(awk 'BEGIN { FS = "\"" } !/^[ \t]*[{}]/ { print $(NF - 1); exit }' modal-login/temp-data/userData.json)
    echo -e "\n${CYAN}${BOLD}[✓] ORG_ID has been set to: ${BOLD}$ORG_ID\n${NC}"
else
    cd modal-login

    echo -e "\n${CYAN}${BOLD}[✓] Installing dependencies with npm. This may take a few minutes, depending on your internet speed...${NC}"
    npm install --legacy-peer-deps
    
    echo -e "\n${CYAN}${BOLD}[✓] Starting the development server...${NC}"
    if ! command -v ss &>/dev/null; then
      echo -e "${YELLOW}[!] 'ss' not found. Attempting to install 'iproute2'...${NC}"
      if command -v apt &>/dev/null; then
        sudo apt update && sudo apt install -y iproute2
      elif command -v yum &>/dev/null; then
        sudo yum install -y iproute
      elif command -v pacman &>/dev/null; then
        sudo pacman -Sy iproute2
      else
        echo -e "${RED}[✗] Could not install 'ss'. Package manager not found.${NC}"
        exit 1
      fi
    fi
    
    PORT_LINE=$(ss -ltnp | grep ":3000 ")
    if [ -n "$PORT_LINE" ]; then
      PID=$(echo "$PORT_LINE" | grep -oP 'pid=\K[0-9]+')
      if [ -n "$PID" ]; then
        echo -e "${YELLOW}[!] Port 3000 is in use. Killing process: $PID${NC}"
        kill -9 $PID
        sleep 2
      fi
    fi
    
    npm run dev > server.log 2>&1 &
    SERVER_PID=$!
    MAX_WAIT=30  
    
    for ((i = 0; i < MAX_WAIT; i++)); do
        if grep -q "Local:        http://localhost:" server.log; then
            PORT=$(grep "Local:        http://localhost:" server.log | sed -n 's/.*http:\/\/localhost:\([0-9]*\).*/\1/p')
            if [ -n "$PORT" ]; then
                echo -e "${GREEN}${BOLD}[✓] Server is running successfully on port $PORT.${NC}"
                break
            fi
        fi
        sleep 1
    done
    
    if [ $i -eq $MAX_WAIT ]; then
        echo -e "${RED}${BOLD}[✗] Timeout waiting for server to start.${NC}"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi

    echo -e "\n${CYAN}${BOLD}[✓] Detecting system architecture...${NC}"
    ARCH=$(uname -m)
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    if [ "$ARCH" = "x86_64" ]; then
        NGROK_ARCH="amd64"
        CF_ARCH="amd64"
        echo -e "${GREEN}${BOLD}[✓] Detected x86_64 architecture.${NC}"
    elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
        NGROK_ARCH="arm64"
        CF_ARCH="arm64"
        echo -e "${GREEN}${BOLD}[✓] Detected ARM64 architecture.${NC}"
    elif [[ "$ARCH" == arm* ]]; then
        NGROK_ARCH="arm"
        CF_ARCH="arm"
        echo -e "${GREEN}${BOLD}[✓] Detected ARM architecture.${NC}"
    else
        echo -e "${RED}[✗] Unsupported architecture: $ARCH. Please use a supported system.${NC}"
        exit 1
    fi

    check_url() {
        local url=$1
        local max_retries=3
        local retry=0
        
        while [ $retry -lt $max_retries ]; do
            http_code=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
            if [ "$http_code" = "200" ] || [ "$http_code" = "404" ] || [ "$http_code" = "301" ] || [ "$http_code" = "302" ]; then
                return 0
            fi
            retry=$((retry + 1))
            sleep 2
        done
        return 1
    }

    install_localtunnel() {
        if command -v lt >/dev/null 2>&1; then
            echo -e "${GREEN}${BOLD}[✓] Localtunnel is already installed.${NC}"
            return 0
        fi
        echo -e "\n${CYAN}${BOLD}[✓] Installing localtunnel...${NC}"
        npm install -g localtunnel > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}${BOLD}[✓] Localtunnel installed successfully.${NC}"
            return 0
        else
            echo -e "${RED}${BOLD}[✗] Failed to install localtunnel.${NC}"
            return 1
        fi
    }

    install_cloudflared() {
        if command -v cloudflared >/dev/null 2>&1; then
            echo -e "${GREEN}${BOLD}[✓] Cloudflared is already installed.${NC}"
            return 0
        fi
        echo -e "\n${YELLOW}${BOLD}[✓] Installing cloudflared...${NC}"
        CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$CF_ARCH"
        wget -q --show-progress "$CF_URL" -O cloudflared
        if [ $? -ne 0 ]; then
            echo -e "${RED}${BOLD}[✗] Failed to download cloudflared.${NC}"
            return 1
        fi
        chmod +x cloudflared
        sudo mv cloudflared /usr/local/bin/
        if [ $? -ne 0 ]; then
            echo -e "${RED}${BOLD}[✗] Failed to move cloudflared to /usr/local/bin/.${NC}"
            return 1
        fi
        echo -e "${GREEN}${BOLD}[✓] Cloudflared installed successfully.${NC}"
        return 0
    }

    install_ngrok() {
        if command -v ngrok >/dev/null 2>&1; then
            echo -e "${GREEN}${BOLD}[✓] ngrok is already installed.${NC}"
            return 0
        fi
        echo -e "${YELLOW}${BOLD}[✓] Installing ngrok...${NC}"
        NGROK_URL="https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-$OS-$NGROK_ARCH.tgz"
        wget -q --show-progress "$NGROK_URL" -O ngrok.tgz
        if [ $? -ne 0 ]; then
            echo -e "${RED}${BOLD}[✗] Failed to download ngrok.${NC}"
            return 1
        fi
        tar -xzf ngrok.tgz
        if [ $? -ne 0 ]; then
            echo -e "${RED}${BOLD}[✗] Failed to extract ngrok.${NC}"
            rm ngrok.tgz
            return 1
        fi
        sudo mv ngrok /usr/local/bin/
        if [ $? -ne 0 ]; then
            echo -e "${RED}${BOLD}[✗] Failed to move ngrok to /usr/local/bin/.${NC}"
            rm ngrok.tgz
            return 1
        fi
        rm ngrok.tgz
        echo -e "${GREEN}${BOLD}[✓] ngrok installed successfully.${NC}"
        return 0
    }

    try_localtunnel() {
        echo -e "\n${CYAN}${BOLD}[✓] Trying localtunnel...${NC}"
        if install_localtunnel; then
            echo -e "\n${CYAN}${BOLD}[✓] Starting localtunnel on port $PORT...${NC}"
            TUNNEL_TYPE="localtunnel"
            lt --port $PORT > localtunnel_output.log 2>&1 &
            TUNNEL_PID=$!
            
            sleep 5
            URL=$(grep -o "https://[^ ]*" localtunnel_output.log | head -n1)
            
            if [ -n "$URL" ]; then
                PASS=$(curl -s https://loca.lt/mytunnelpassword)
                FORWARDING_URL="$URL"
                echo -e "${GREEN}${BOLD}[✓] Success! Please visit this website : ${YELLOW}${BOLD}${URL}${GREEN}${BOLD} and then enter this password : ${YELLOW}${BOLD}${PASS}${GREEN}${BOLD} to access the website and then log in using your email.${NC}"
                return 0
            else
                echo -e "${RED}${BOLD}[✗] Failed to get localtunnel URL.${NC}"
                kill $TUNNEL_PID 2>/dev/null || true
            fi
        fi
        return 1
    }

    try_cloudflared() {
      echo -e "\n${CYAN}${BOLD}[✓] Trying cloudflared...${NC}"
      
      # Khởi động tunnel cloudflared
      echo -e "${CYAN}${BOLD}[✓] Starting cloudflared tunnel on http://localhost:$PORT...${NC}"
      TUNNEL_TYPE="cloudflared"
      cloudflared tunnel --url http://localhost:$PORT > cloudflared_output.log 2>&1 &
      TUNNEL_PID=$!
      
      # Chờ URL xuất hiện, tối đa 30 giây
      counter=0
      MAX_WAIT=30
      echo -e "${CYAN}${BOLD}[✓] Waiting for cloudflared URL (up to $MAX_WAIT seconds)...${NC}"
      while [ $counter -lt $MAX_WAIT ]; do
          CLOUDFLARED_URL=$(grep -o 'https://[a-zA-Z0-9.-]*\.trycloudflare\.com' cloudflared_output.log | head -n1)
          if [ -n "$CLOUDFLARED_URL" ]; then
              echo -e "${GREEN}${BOLD}[✓] Cloudflared tunnel started: $CLOUDFLARED_URL${NC}"
              FORWARDING_URL="$CLOUDFLARED_URL"
              return 0
          fi
          sleep 1
          counter=$((counter + 1))
      done
  
      # Nếu không lấy được URL
      echo -e "${RED}${BOLD}[✗] Failed to retrieve cloudflared URL after $MAX_WAIT seconds.${NC}"
      echo -e "${RED}${BOLD}[✗] Check cloudflared_output.log for details:${NC}"
      [ -f cloudflared_output.log ] && cat cloudflared_output.log
      kill $TUNNEL_PID 2>/dev/null || true
      return 1
  }
  
      get_ngrok_url_method2() {
          local try_port
          local url=""
          for try_port in $(seq 4040 4045); do
              local response=$(curl -s "http://localhost:$try_port/api/tunnels" 2>/dev/null)
              if [ -n "$response" ]; then
                  url=$(echo "$response" | grep -o '"public_url":"https://[^"]*' | head -n1 | cut -d'"' -f4)
                  if [ -n "$url" ]; then
                      break
                  fi
              fi
          done
          echo "$url"
      }

    get_ngrok_url_method3() {
        local url=$(grep -o "Forwarding.*https://[^ ]*" ngrok_output.log 2>/dev/null | grep -o "https://[^ ]*" | head -n1)
        echo "$url"
    }

    try_ngrok() {
        echo -e "\n${CYAN}${BOLD}[✓] Trying ngrok...${NC}"
        if install_ngrok; then
            TUNNEL_TYPE="ngrok"
            while true; do
                echo -e "\n${YELLOW}${BOLD}To get your authtoken:${NC}"
                echo "1. Sign up or log in at https://dashboard.ngrok.com"
                echo "2. Go to 'Your Authtoken' section: https://dashboard.ngrok.com/get-started/your-authtoken"
                echo "3. Click on the eye icon to reveal your ngrok auth token"
                echo "4. Copy that auth token and paste it in the prompt below"
                echo -e "\n${BOLD}Please enter your ngrok authtoken:${NC}"
                read -p "> " NGROK_TOKEN
            
                if [ -z "$NGROK_TOKEN" ]; then
                    echo -e "${RED}${BOLD}[✗] No token provided. Please enter a valid token.${NC}"
                    continue
                fi
                pkill -f ngrok || true
                sleep 2
            
                ngrok authtoken "$NGROK_TOKEN" 2>/dev/null
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}${BOLD}[✓] Successfully authenticated ngrok!${NC}"
                    break
                else
                    echo -e "${RED}[✗] Authentication failed. Please check your token and try again.${NC}"
                fi
            done

            echo -e "\n${CYAN}${BOLD}[✓] Starting ngrok with method 1...${NC}"
            ngrok http "$PORT" --log=stdout --log-format=json > ngrok_output.log 2>&1 &
            TUNNEL_PID=$!
            sleep 5
            
            NGROK_URL=$(get_ngrok_url_method1)
            if [ -n "$NGROK_URL" ]; then
                FORWARDING_URL="$NGROK_URL"
                return 0
            else
                echo -e "${RED}${BOLD}[✗] Failed to get ngrok URL (method 1).${NC}"
                kill $TUNNEL_PID 2>/dev/null || true
            fi

            echo -e "\n${CYAN}${BOLD}[✓] Starting ngrok with method 2...${NC}"
            ngrok http "$PORT" > ngrok_output.log 2>&1 &
            TUNNEL_PID=$!
            sleep 5
            
            NGROK_URL=$(get_ngrok_url_method2)
            if [ -n "$NGROK_URL" ]; then
                FORWARDING_URL="$NGROK_URL"
                return 0
            else
                echo -e "${RED}${BOLD}[✗] Failed to get ngrok URL (method 2).${NC}"
                kill $TUNNEL_PID 2>/dev/null || true
            fi

            echo -e "\n${CYAN}${BOLD}[✓] Starting ngrok with method 3...${NC}"
            ngrok http "$PORT" --log=stdout > ngrok_output.log 2>&1 &
            TUNNEL_PID=$!
            sleep 5
            
            NGROK_URL=$(get_ngrok_url_method3)
            if [ -n "$NGROK_URL" ]; then
                FORWARDING_URL="$NGROK_URL"
                return 0
            else
                echo -e "${RED}${BOLD}[✗] Failed to get ngrok URL (method 3).${NC}"
                kill $TUNNEL_PID 2>/dev/null || true
            fi
        fi
        return 1
    }

    start_tunnel() {
        if try_localtunnel; then
            return 0
        fi
        
        if try_cloudflared; then
            return 0
        fi
        
        if try_ngrok; then
            return 0
        fi
        return 1
    }

    start_tunnel
    if [ $? -eq 0 ]; then
        if [ "$TUNNEL_TYPE" != "localtunnel" ]; then
            echo -e "${GREEN}${BOLD}[✓] Success! Please visit this website and log in using your email:${NC} ${CYAN}${BOLD}${FORWARDING_URL}${NC}"
        fi
    else
        echo -e "\n${BLUE}${BOLD}[✓] Don't worry, you can use this manual method. Please follow these instructions:${NC}"
        echo "1. Open this same WSL/VPS or GPU server on another tab"
        echo "2. Paste this command into this terminal: ngrok http $PORT"
        echo "3. It will show a link similar to this: https://xxxx.ngrok-free.app"
        echo "4. Visit this website and login using your email, this website may take 30 sec to load."
        echo "5. Now go back to the previous tab, you will see everything will run fine"
    fi

    cd ..

    echo -e "\n${CYAN}${BOLD}[↻] Waiting for you to complete the login process...${NC}"
    while [ ! -f "modal-login/temp-data/userData.json" ]; do
        sleep 3
    done
    
    echo -e "${GREEN}${BOLD}[✓] Success! The userData.json file has been created. Proceeding with remaining setups...${NC}"
    rm -f server.log localtunnel_output.log cloudflared_output.log ngrok_output.log

    ORG_ID=$(awk 'BEGIN { FS = "\"" } !/^[ \t]*[{}]/ { print $(NF - 1); exit }' modal-login/temp-data/userData.json)
    echo -e "\n${CYAN}${BOLD}[✓] ORG_ID has been set to: $ORG_ID\n${NC}"

    echo -e "${CYAN}${BOLD}[✓] Waiting for API key to become activated...${NC}"
    while true; do
        STATUS=$(curl -s "http://localhost:3000/api/get-api-key-status?orgId=$ORG_ID")
        if [[ "$STATUS" == "activated" ]]; then
            echo -e "${GREEN}${BOLD}[✓] Success! API key is activated! Proceeding...\n${NC}"
            break
        else
            echo "[↻] Waiting for API key to be activated..."
            sleep 5
        fi
    done

    ENV_FILE="$ROOT"/modal-login/.env
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS version
        sed -i '' "3s/.*/SMART_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$ENV_FILE"
    else
        # Linux version
        sed -i "3s/.*/SMART_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$ENV_FILE"
    fi
fi

echo -e "${CYAN}${BOLD}[✓] Setting up Python virtual environment...${NC}"
python3 -m venv .venv && . .venv/bin/activate && \
echo -e "${GREEN}${BOLD}[✓] Python virtual environment set up successfully.${NC}" || \
echo -e "${RED}${BOLD}[✗] Failed to set up virtual environment.${NC}"

if [ -z "$CONFIG_PATH" ]; then
    if command -v nvidia-smi &> /dev/null || [ -d "/proc/driver/nvidia" ]; then
        echo -e "${GREEN}${BOLD}[✓] GPU detected${NC}"
        
        # Here was the problematic break statement - removing it and fixing logic
        case "$PARAM_B" in
            32 | 72) 
                CONFIG_PATH="$ROOT/hivemind_exp/configs/gpu/grpo-qwen-2.5-${PARAM_B}b-bnb-4bit-deepseek-r1.yaml"
                ;;
            0.5 | 1.5 | 7) 
                CONFIG_PATH="$ROOT/hivemind_exp/configs/gpu/grpo-qwen-2.5-${PARAM_B}b-deepseek-r1.yaml"
                ;;
            *)  
                echo ">>> Parameter size not recognized. Defaulting to 0.5b."
                CONFIG_PATH="$ROOT/hivemind_exp/configs/gpu/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
                ;;
        esac
        
        if [ "$USE_BIG_SWARM" = true ]; then
            GAME="dapo"
        else
            GAME="gsm8k"
        fi
        echo -e "${GREEN}${BOLD}[✓] Config file : ${BOLD}$CONFIG_PATH\n${NC}"
        echo -e "${CYAN}${BOLD}[✓] Installing GPU-specific requirements, may take few mins depending on your internet speed...${NC}"
        pip install -r "$ROOT"/requirements-gpu.txt
        pip install flash-attn --no-build-isolation
    else
        echo -e "${YELLOW}${BOLD}[✓] No GPU detected, using CPU configuration${NC}"
        pip install -r "$ROOT"/requirements-cpu.txt
        CONFIG_PATH="$ROOT/hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
        GAME="gsm8k"
        echo -e "${CYAN}${BOLD}[✓] Config file : ${BOLD}$CONFIG_PATH\n${NC}"
    fi
fi


if [ -n "${HF_TOKEN}" ]; then
    HUGGINGFACE_ACCESS_TOKEN=${HF_TOKEN}
else
    echo -e "\n${CYAN}Would you like to push models you train in the RL swarm to the Hugging Face Hub? [y/N]${NC}"
    echo -n "> "
    read yn
    yn=${yn:-N}
    case "$yn" in
        [Yy]* ) 
            read -p "Enter your Hugging Face access token: " HUGGINGFACE_ACCESS_TOKEN
            ;;
        [Nn]* ) 
            HUGGINGFACE_ACCESS_TOKEN="None"
            ;;
        * ) 
            echo -e "${YELLOW}>>> No answer was given, so NO models will be pushed to the Hugging Face Hub.${NC}"
            HUGGINGFACE_ACCESS_TOKEN="None"
            ;;
    esac
fi

echo -e "\n${CYAN}${BOLD}This is your preferred ENV for this training session :\n${NC}"

print_env() {
    local name=$1
    local value=${!name}
    value=${value:-"Not Available"}
    echo -e "+ ${PURPLE}${BOLD}${name}${NC} : ${value}"
}

print_env "HF_TOKEN"
print_env "ORG_ID"
print_env "IDENTITY_PATH"
print_env "SWARM_CONTRACT"
print_env "CONFIG_PATH"
print_env "GAME"
print_env "PUB_MULTI_ADDRS"
print_env "PEER_MULTI_ADDRS"
print_env "HOST_MULTI_ADDRS"

sleep 5

echo -e "\n${GREEN}${BOLD}[✓] Good luck in the swarm! Your training session is about to begin.\n${NC}"
[ "$(uname)" = "Darwin" ] && sed -i '' -E 's/(startup_timeout: *float *= *)[0-9.]+/\1120/' $(python3 -c "import hivemind.p2p.p2p_daemon as m; print(m.__file__)") || sed -i -E 's/(startup_timeout: *float *= *)[0-9.]+/\1120/' $(python3 -c "import hivemind.p2p.p2p_daemon as m; print(m.__file__)")
[ "$(uname)" = "Darwin" ] && sed -i '' -e '/bootstrap_timeout: Optional\[float\] = None/s//bootstrap_timeout: float = 120/' $(python3 -c 'import hivemind.dht.node as m; print(m.__file__)') || sed -i -e '/bootstrap_timeout: Optional\[float\] = None/s//bootstrap_timeout: float = 120/' $(python3 -c 'import hivemind.dht.node as m; print(m.__file__)')
if [ -n "$ORG_ID" ]; then
    python -m hivemind_exp.gsm8k.train_single_gpu \
        --hf_token "$HUGGINGFACE_ACCESS_TOKEN" \
        --identity_path "$IDENTITY_PATH" \
        --modal_org_id "$ORG_ID" \
        --contract_address "$SWARM_CONTRACT" \
        --config "$CONFIG_PATH" \
        --game "$GAME"
else
    python -m hivemind_exp.gsm8k.train_single_gpu \
        --hf_token "$HUGGINGFACE_ACCESS_TOKEN" \
        --identity_path "$IDENTITY_PATH" \
        --public_maddr "$PUB_MULTI_ADDRS" \
        --initial_peers "$PEER_MULTI_ADDRS" \
        --host_maddr "$HOST_MULTI_ADDRS" \
        --config "$CONFIG_PATH" \
        --game "$GAME"
fi

wait
Beta
0 / 10
used queries
1
