### Response Templates (feature: `prompt-templates`)

Tool **responses** can also be rendered through templates. The tool's return
value (`T: Serialize`) is serialized into a template context, rendered, and
returned as `ToolOutput`. The original struct is attached as metadata for
programmatic access.

`tools/weather_response.tmpl.md`:

```markdown
---
name: weather_response
params:
  - city = str
  - temp_f = int
  - condition = str
  - humidity = int
---

🌤️ **Weather for {{ city }}**

- **Temperature**: {{ temp_f }}°F
- **Condition**: {{ condition }}
- **Humidity**: {{ humidity }}%
```

```rust
use llm_tool::{llm_tool, ToolError, ToolRegistry, ToolContext};
use serde::Serialize;

#[derive(Serialize)]
struct WeatherResponse {
    city: String,
    temp_f: i64,
    condition: String,
    humidity: i64,
}

/// Get the weather for a city.
#[llm_tool(
    response_file = "tools/weather_response.tmpl.md"
)]
fn get_weather_templated(
    /// The city to get weather for.
    city: String,
) -> Result<WeatherResponse, ToolError> {
    Ok(WeatherResponse {
        city,
        temp_f: 72,
        condition: "Sunny".into(),
        humidity: 45,
    })
}

# futures::executor::block_on(async {
let registry = ToolRegistry::new().with_tool(GetWeatherTemplated);
let ctx = ToolContext::new(None);

let output = registry
    .dispatch("get_weather_templated", serde_json::json!({"city": "Seattle"}), &ctx)
    .await
    .unwrap();

// The rendered template is the tool output the model sees:
assert!(output.content().contains("Weather for Seattle"));
assert!(output.content().contains("72°F"));

// The struct fields are also available as metadata for programmatic use:
assert_eq!(output.metadata()["city"], "Seattle");
assert_eq!(output.metadata()["temp_f"], 72);
# });
```

| Behaviour        | Detail                                                                          |
| ---------------- | ------------------------------------------------------------------------------- |
| Context building | Struct is serialized via `Context::from_serialize` — all fields become vars.    |
| Template caching | Parsed once at startup via `LazyLock`, zero overhead on subsequent calls.       |
| Metadata         | The full struct is attached as `ToolOutput` metadata for hooks and logging.      |
| Compile-time     | Missing template files and syntax errors are caught during `cargo build`.       |
| Combinable       | Works with `/// doc comments` or `prompt_file = "..."` for the tool description.|
