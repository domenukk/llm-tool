---
name: parameterized_desc
description: Template with compile-time params
types:
  - ApiContext = struct(api_version = str, env_name = str)

allow_unused: true
params:
  - context = ApiContext
---

Fetch weather data using API {{ context.api_version }} in {{ context.env_name }} environment.
