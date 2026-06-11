use llm_tool::llm_tool;

struct MyStruct;

impl MyStruct {
    /// Does something with self.
    #[llm_tool]
    fn with_self(&self) -> Result<String, String> {
        Ok("hello".to_string())
    }
}

fn main() {}
