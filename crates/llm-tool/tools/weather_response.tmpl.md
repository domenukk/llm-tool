---
name: weather_response
description: Response template for the get_weather tool
params:
  - city = untrusted str
  - temp_f = int
  - condition = untrusted str
  - humidity = int
---

🌤️ **Weather for {{ city }}**

- **Temperature**: {{ temp_f }}°F
- **Condition**: {{ condition }}
- **Humidity**: {{ humidity }}%
