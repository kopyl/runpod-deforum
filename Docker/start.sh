#!/bin/bash
echo "Container Started"
echo "Starting api"
python -m cog.server.http & cd ./
echo "starting worker"
python -u handler.py
