---
name: search_response
types:
  - SearchResultItem = struct(title = untrusted str, score = int)

allow_unused: true
params:
  - query = untrusted str
  - results = list(SearchResultItem)
  - total = int
---

Search results for "{{ query }}" ({{ total }} found):

> {% for r in results %}

- {{ r.title }} (score: {{ r.score }})

> {% /for %}
