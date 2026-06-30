---
name: search_response
types:
  - SearchResultItem = struct(title = str, score = int)

allow_unused: true
params:
  - query = str
  - results = list(SearchResultItem)
  - total = int
---

Search results for "{{ query }}" ({{ total }} found):

> {% for r in results %}

- {{ r.title }} (score: {{ r.score }})

  > {% /for %}
