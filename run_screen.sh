#!/bin/bash
echo "Session name: $1"
echo "Command: ${@:2}"
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Time: $timestamp"
mkdir -p logs/$1_$timestamp
echo "Saved into logs/$1_$timestamp/ directory"
screen -S $1 -L -Logfile logs/$1_$timestamp/screenlog.log  -d -m "${@:2}"
