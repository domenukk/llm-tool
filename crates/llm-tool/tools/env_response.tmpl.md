---
name: env_response
description: Response with env

env:
  - SERVICE_NAME = str

params:
  - result = str
  - count = int
---

[{{ SERVICE_NAME }}] Found {{ count }} results: {{ result }}
