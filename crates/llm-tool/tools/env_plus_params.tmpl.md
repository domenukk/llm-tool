---
name: env_plus_params
description: Template with both env and params

env:
  - DEPLOYMENT_ENV = str

params:
  - version = str
---

Version {{ version }} running in {{ DEPLOYMENT_ENV }}.
