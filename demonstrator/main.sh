#!/bin/bash
# start dash app
uvicorn app:app --host 0.0.0.0 --port 8051
status=$?
if [ $status -ne 0 ]; then
  echo "Failed to start dash app: $status"
  exit $status
fi