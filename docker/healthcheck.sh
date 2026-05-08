#!/bin/sh
# Healthcheck for the API container
# Used by Docker to determine if the container is ready
curl -sf http://localhost:8000/health || exit 1
