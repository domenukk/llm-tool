---
name: parameterized_desc
description: Template with compile-time params
params:
  - api_version = str
  - env_name = str
---

Fetch weather data using API {{ api_version }} in {{ env_name }} environment.
