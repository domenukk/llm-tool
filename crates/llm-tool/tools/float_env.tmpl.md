---
name: float_env
description: Template with float env variable
env:
  - THRESHOLD = float
  - MIN_SCORE = float := 0.5
---

Filter results above {{ THRESHOLD }} (minimum score: {{ MIN_SCORE }}).
