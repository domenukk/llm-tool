---
name: multi_env
description: Template with multiple env variables
env:
  - SERVICE_NAME = str
  - REGION = str
  - MAX_CONNECTIONS = int := 100
  - DEBUG_MODE = bool := false
---

Service {{ SERVICE_NAME }} in {{ REGION }} with {{ MAX_CONNECTIONS }} connections (debug={{ DEBUG_MODE }}).
