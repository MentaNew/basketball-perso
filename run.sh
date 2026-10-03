#!/usr/bin/env bash
cd "$(dirname "$0")"
PYTHONPATH=. streamlit run app/Home.py "$@"
