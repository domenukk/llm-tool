use super::*;

#[tokio::test]
async fn test_prompts_list() {
    let server = test_server_with_prompts_and_resources();
    let line = r#"{"jsonrpc":"2.0","id":1,"method":"prompts/list"}"#;
    let resp = server.handle_request(line).await;
    assert_eq!(resp.id, Some(serde_json::json!(1)));
    let val = serde_json::to_value(resp.result.unwrap()).unwrap();
    let prompts = val["prompts"].as_array().unwrap();
    assert_eq!(prompts.len(), 1);
    assert_eq!(prompts[0]["name"], "review_prompt");
}

#[tokio::test]
async fn test_prompts_get() {
    let server = test_server_with_prompts_and_resources();
    let line = r#"{"jsonrpc":"2.0","id":2,"method":"prompts/get","params":{"name":"review_prompt","arguments":{"lang":"Rust","code":"fn main() {}"}}}"#;
    let resp = server.handle_request(line).await;
    assert_eq!(resp.id, Some(serde_json::json!(2)));
    let val = serde_json::to_value(resp.result.unwrap()).unwrap();
    let messages = val["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"]["type"], "text");
    assert!(
        messages[0]["content"]["text"]
            .as_str()
            .unwrap()
            .contains("Review code for Rust")
    );
}

#[tokio::test]
async fn test_resources_list() {
    let server = test_server_with_prompts_and_resources();
    let line = r#"{"jsonrpc":"2.0","id":3,"method":"resources/list"}"#;
    let resp = server.handle_request(line).await;
    assert_eq!(resp.id, Some(serde_json::json!(3)));
    let val = serde_json::to_value(resp.result.unwrap()).unwrap();
    let resources = val["resources"].as_array().unwrap();
    assert_eq!(resources.len(), 1);
    assert_eq!(resources[0]["name"], "get_config");
    assert_eq!(resources[0]["uri"], "file:///config/{app}.json");
    assert_eq!(resources[0]["mimeType"], "application/json");
}

#[tokio::test]
async fn test_resources_templates_list() {
    let server = test_server_with_prompts_and_resources();
    let line = r#"{"jsonrpc":"2.0","id":4,"method":"resources/templates/list"}"#;
    let resp = server.handle_request(line).await;
    assert_eq!(resp.id, Some(serde_json::json!(4)));
    let val = serde_json::to_value(resp.result.unwrap()).unwrap();
    let templates = val["resourceTemplates"].as_array().unwrap();
    assert_eq!(templates.len(), 1);
    assert_eq!(templates[0]["name"], "get_config");
    assert_eq!(templates[0]["uriTemplate"], "file:///config/{app}.json");
}

#[tokio::test]
async fn test_resources_read() {
    let server = test_server_with_prompts_and_resources();
    let line = r#"{"jsonrpc":"2.0","id":5,"method":"resources/read","params":{"uri":"file:///config/test-app.json"}}"#;
    let resp = server.handle_request(line).await;
    assert_eq!(resp.id, Some(serde_json::json!(5)));
    let val = serde_json::to_value(resp.result.unwrap()).unwrap();
    let contents = val["contents"].as_array().unwrap();
    assert_eq!(contents.len(), 1);
    assert_eq!(contents[0]["uri"], "file:///config/test-app.json");
    assert_eq!(contents[0]["mimeType"], "application/json");
    assert!(contents[0]["text"].as_str().unwrap().contains("test-app"));
}

#[tokio::test]
async fn rmcp_client_prompts_resources_integration_test() {
    let server = test_server_with_prompts_and_resources();
    let (client_io, server_io) = tokio::io::duplex(8192);
    let (server_r, server_w) = tokio::io::split(server_io);

    tokio::spawn(async move {
        let mut reader = tokio::io::BufReader::new(server_r);
        // NOLINT: test background task — server result unused; test controls lifecycle
        let _ = server.run_async(&mut reader, server_w).await;
    });

    let mut client = rmcp::service::serve_client((), client_io)
        .await
        .expect("rmcp client handshake failed");

    // 1. Verify list_all_prompts
    let prompts = client
        .list_all_prompts()
        .await
        .expect("list prompts failed");
    assert_eq!(prompts.len(), 1);
    assert_eq!(prompts[0].name, "review_prompt");

    // 2. Verify get_prompt
    let get_res = client
        .get_prompt(
            rmcp::model::GetPromptRequestParams::new("review_prompt").with_arguments(
                serde_json::json!({"lang": "TypeScript", "code": "const x = 1;"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
        )
        .await
        .expect("get prompt failed");
    assert_eq!(get_res.messages.len(), 1);

    // 3. Verify list_all_resources
    let resources = client
        .list_all_resources()
        .await
        .expect("list resources failed");
    assert_eq!(resources.len(), 1);
    assert_eq!(resources[0].name, "get_config");

    // 4. Verify read_resource
    let read_res = client
        .read_resource(rmcp::model::ReadResourceRequestParams::new(
            "file:///config/test-service.json",
        ))
        .await
        .expect("read resource failed");
    assert_eq!(read_res.contents.len(), 1);

    // NOLINT: test cleanup — close errors are non-fatal
    let _ = client.close().await;
}
