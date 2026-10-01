#!/bin/bash
set -u
PID=1262062
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
COUNT=$(find /root/FreqOrtho-SR2/full100_out -maxdepth 1 -type f -name 'case*.png' | wc -l)
echo "FREQ_DONE count=$COUNT" > /root/FreqOrtho-SR2/watch_freqortho_full100.log
if [ "$COUNT" -eq 100 ]; then
  cd /root/rivermind-data
  nohup /opt/conda/bin/python -u run_teacher_freqortho_full100.py --cases $(seq 1 100) --out /root/FreqOrtho-SR2/full100_teacher --teacher /root/FreqOrtho-SR2/full100_out >> /root/FreqOrtho-SR2/watch_freqortho_full100.log 2>&1 &
fi
