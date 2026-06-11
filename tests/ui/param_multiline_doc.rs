use llm_tool::llm_tool;
use llm_tool::definition_of;

/// Tool with multiline parameter documentation.
#[llm_tool]
fn multiline_param_tool(
    /// First line of parameter doc.
    ///
    /// Second line of parameter doc.
    x: i64,
) -> Result<String, String> {
    Ok(format!("{x}"))
}

fn main() {
    let def = definition_of(&MultilineParamTool).unwrap();
    let schema = def.parameter_schema;
    
    // The JSON Schema structure for properties:
    // {
    //   "type": "object",
    //   "properties": {
    //     "x": {
    //       "description": "First line of parameter doc.\n\nSecond line of parameter doc.",
    //       "type": "integer",
    //       ...
    //     }
    //   },
    //   ...
    // }
    let x_desc = schema
        .get("properties")
        .and_then(|p| p.get("x"))
        .and_then(|x| x.get("description"))
        .and_then(|d| d.as_str())
        .unwrap();
        
    assert!(
        x_desc.contains('\n'),
        "Parameter description should preserve newlines, but was: {:?}",
        x_desc
    );
    assert_eq!(
        x_desc,
        "First line of parameter doc.\n\nSecond line of parameter doc."
    );
}
