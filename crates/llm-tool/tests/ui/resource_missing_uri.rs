use llm_tool::llm_resource;

/// A resource without the required `uri`.
#[llm_resource(description = "no uri here")]
fn my_resource() -> String {
    "hello".to_string()
}

fn main() {}
