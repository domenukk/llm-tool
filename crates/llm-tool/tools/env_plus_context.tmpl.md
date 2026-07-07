---
name: env_plus_context
description: Template with both env and context params

env:
  - CLUSTER = str

params:
  - region = str
---

Deployed to {{ region }} on cluster {{ CLUSTER }}.
