---
name: search_response
params:
  - query = str
  - results = list<title = str, score = int>
  - total = int
---

Search results for "{{ query }}" ({{ total }} found):

> {% for r in results %}

- {{ r.title }} (score: {{ r.score }})

  > {% /for %}
