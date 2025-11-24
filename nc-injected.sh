#!/bin/bash

# Script to keep netcat running on port 4444
# It will automatically restart nc if it dies

PORT=4444

echo "Starting netcat auto-restart script on port $PORT"
echo "Press Ctrl+C to stop the script"

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting nc on port $PORT..."

  # Start netcat in listen mode
  killall nc
  echo "exit" | nc -l -p $PORT

  # If nc exits, we'll reach here
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] nc died, restarting in 2 seconds..."
  sleep 2
done