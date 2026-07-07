---
name: env_desc
description: Template with env variables

env:
  - API_VERSION = str
  - MAX_RETRIES = int := 3
---

Fetch data using API {{ API_VERSION }} with up to {{ MAX_RETRIES }} retries.
