---
name: dynamic_tool
description: A tool with a dynamic description
types:
  - ApiContext = struct(api_version = str, env_name = str)
params:
  - context = ApiContext
---

Perform weather checks. (Running on API {{ context.api_version }} in {{ context.env_name }} environment).
