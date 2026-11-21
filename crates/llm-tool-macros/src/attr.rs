//! Attribute parsing and compile-time validation for `#[llm_tool]`, `#[llm_prompt]`, and `#[llm_resource]`.

use syn::{Ident, LitStr};

use crate::helpers;

/// Maximum byte length allowed for an XML boundary tag name in `OutputSanitization`.
const MAX_SANITIZE_TAG_LEN: usize = 128;

/// Maximum byte length allowed for a custom boundary notice in `OutputSanitization`.
const MAX_SANITIZE_NOTICE_LEN: usize = 2048;

/// Reserved tag name forbidden for custom sanitize tags (`<surrogate:kind:nonce>`).
const RESERVED_SURROGATE_TAG: &str = "surrogate";

pub const ATTR_NAME: &str = "name";
pub const ATTR_DESCRIPTION: &str = "description";
pub const ATTR_DESCRIPTION_FILE: &str = "description_file";
const ATTR_RESPONSE_FILE: &str = "response_file";
const ATTR_RESPONSE: &str = "response";
pub const ATTR_PARAMS: &str = "params";
pub const ATTR_CONTEXT: &str = "context";
pub const ATTR_ENV: &str = "env";
pub const ATTR_IDEMPOTENT: &str = "idempotent";
pub const ATTR_EFFECT: &str = "effect";
pub const ATTR_PROVENANCE: &str = "provenance";
pub const ATTR_SANITIZE: &str = "sanitize";
pub const ATTR_QUARANTINE: &str = "quarantine";
pub const ATTR_SANITIZATION: &str = "sanitization";
const ATTR_SANITIZE_TAG_ARG: &str = "tag";
const ATTR_SANITIZE_NOTICE_ARG: &str = "notice";

/// Returns `true` if `name` is a valid non-empty ASCII XML 1.0 `NCName` (`[A-Za-z_][A-Za-z0-9_.-]*`).
const fn is_valid_xml_ncname(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let first = bytes[0];
    if !(first.is_ascii_alphabetic() || first == b'_') {
        return false;
    }
    let mut idx = 1;
    while idx < bytes.len() {
        let b = bytes[idx];
        if !(b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.') {
            return false;
        }
        idx += 1;
    }
    true
}

fn validate_sanitize_tag_lit(lit: &LitStr) -> syn::Result<()> {
    let val = lit.value();
    if val.len() > MAX_SANITIZE_TAG_LEN {
        return Err(syn::Error::new(
            lit.span(),
            format!(
                "sanitize tag length ({} bytes) exceeds maximum of {MAX_SANITIZE_TAG_LEN} bytes",
                val.len()
            ),
        ));
    }
    if val.eq_ignore_ascii_case(RESERVED_SURROGATE_TAG) {
        return Err(syn::Error::new(
            lit.span(),
            format!(
                "sanitize tag `{val}` is reserved for `<surrogate:kind:nonce>` handles; choose a distinct XML tag name"
            ),
        ));
    }
    if !is_valid_xml_ncname(&val) {
        return Err(syn::Error::new(
            lit.span(),
            format!(
                "sanitize tag `{val}` is not a valid ASCII XML NCName (must match `[A-Za-z_][A-Za-z0-9_.-]*`)"
            ),
        ));
    }
    Ok(())
}

fn validate_sanitize_notice_lit(lit: &LitStr) -> syn::Result<()> {
    let val = lit.value();
    if val.len() > MAX_SANITIZE_NOTICE_LEN {
        return Err(syn::Error::new(
            lit.span(),
            format!(
                "sanitize notice length ({} bytes) exceeds maximum of {MAX_SANITIZE_NOTICE_LEN} bytes",
                val.len()
            ),
        ));
    }
    Ok(())
}

/// Parsed `#[llm_tool(...)]` / `#[llm_prompt(...)]` attribute.
pub struct ToolAttr {
    /// Explicit wire name override (`name = "..."`).
    pub name_override: Option<LitStr>,
    /// Inline description string (mutually exclusive with `description_file_path`).
    pub description_inline: Option<LitStr>,
    /// Path to a `.tmpl.md` file (mutually exclusive with `description_inline`).
    pub description_file_path: Option<LitStr>,
    /// Path to a response `.tmpl.md` file for auto-rendering tool output.
    pub response_file_path: Option<LitStr>,
    /// Inline response template string (mutually exclusive with `response_file_path`).
    pub response_inline: Option<LitStr>,
    /// Compile-time key-value pairs for template rendering.
    pub inline_params: Vec<(Ident, LitStr)>,
    /// Compile-time environment variables for `env:` frontmatter declarations.
    pub env_vars: Vec<(Ident, syn::Lit)>,
    /// Runtime context function (mutually exclusive with `inline_params`).
    pub context_fn: Option<syn::Path>,
    pub idempotent: bool,
    pub effect: MacroToolEffect,
    /// Span if `effect` or `idempotent` was explicitly specified (rejected on prompts/resources).
    pub explicit_effect_span: Option<proc_macro2::Span>,
    pub provenance: helpers::MacroToolProvenance,
    pub sanitization: MacroOutputSanitization,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum MacroToolEffect {
    ReadOnly,
    Mutating,
    Destructive,
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub enum MacroOutputSanitization {
    None,
    #[default]
    SanitizeTokens,
    Quarantine,
    SanitizeTagged(syn::LitStr),
    SanitizeWithNotice {
        tag: syn::LitStr,
        notice: syn::LitStr,
    },
}

fn parse_effect_str(s: &str, span: proc_macro2::Span) -> syn::Result<MacroToolEffect> {
    match s {
        "read_only" | "ReadOnly" => Ok(MacroToolEffect::ReadOnly),
        "mutating" | "Mutating" => Ok(MacroToolEffect::Mutating),
        "destructive" | "Destructive" => Ok(MacroToolEffect::Destructive),
        _ => Err(syn::Error::new(
            span,
            "expected `read_only`, `mutating`, or `destructive`",
        )),
    }
}

fn parse_sanitization_str(
    s: &str,
    span: proc_macro2::Span,
) -> syn::Result<MacroOutputSanitization> {
    match s {
        "none" | "None" => Ok(MacroOutputSanitization::None),
        "sanitize_tokens" | "sanitize" | "SanitizeTokens" => {
            Ok(MacroOutputSanitization::SanitizeTokens)
        }
        "quarantine" | "Quarantine" => Ok(MacroOutputSanitization::Quarantine),
        _ => Err(syn::Error::new(
            span,
            "expected `none`, `sanitize_tokens`, or `quarantine`",
        )),
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ToolAttrKey {
    Name,
    Description,
    DescriptionFile,
    ResponseFile,
    Response,
    Params,
    Env,
    Context,
    Idempotent,
    Effect,
    Provenance,
    Sanitize,
    Quarantine,
    Sanitization,
}

impl ToolAttrKey {
    const ALL: &'static [Self] = &[
        Self::Name,
        Self::Description,
        Self::DescriptionFile,
        Self::Response,
        Self::ResponseFile,
        Self::Params,
        Self::Env,
        Self::Context,
        Self::Idempotent,
        Self::Effect,
        Self::Provenance,
        Self::Sanitize,
        Self::Quarantine,
        Self::Sanitization,
    ];

    const fn as_str(self) -> &'static str {
        match self {
            Self::Name => ATTR_NAME,
            Self::Description => ATTR_DESCRIPTION,
            Self::DescriptionFile => ATTR_DESCRIPTION_FILE,
            Self::ResponseFile => ATTR_RESPONSE_FILE,
            Self::Response => ATTR_RESPONSE,
            Self::Params => ATTR_PARAMS,
            Self::Env => ATTR_ENV,
            Self::Context => ATTR_CONTEXT,
            Self::Idempotent => ATTR_IDEMPOTENT,
            Self::Effect => ATTR_EFFECT,
            Self::Provenance => ATTR_PROVENANCE,
            Self::Sanitize => ATTR_SANITIZE,
            Self::Quarantine => ATTR_QUARANTINE,
            Self::Sanitization => ATTR_SANITIZATION,
        }
    }

    fn expected_keys_error(span: proc_macro2::Span) -> syn::Error {
        let mut parts: Vec<String> = Self::ALL
            .iter()
            .map(|k| format!("`{}`", k.as_str()))
            .collect();
        let last = parts.pop().expect("ToolAttrKey::ALL is non-empty");
        let formatted = if parts.is_empty() {
            last
        } else {
            format!("{}, or {last}", parts.join(", "))
        };
        syn::Error::new(span, format!("expected {formatted}"))
    }
}

impl TryFrom<&syn::Ident> for ToolAttrKey {
    type Error = syn::Error;

    fn try_from(ident: &syn::Ident) -> Result<Self, Self::Error> {
        let s = ident.to_string();
        for &variant in Self::ALL {
            if s == variant.as_str() {
                return Ok(variant);
            }
        }
        Err(Self::expected_keys_error(ident.span()))
    }
}

#[derive(Default)]
pub struct ToolAttrBuilder {
    name_override: Option<syn::LitStr>,
    description_inline: Option<syn::LitStr>,
    description_file_path: Option<syn::LitStr>,
    response_file_path: Option<syn::LitStr>,
    response_inline: Option<syn::LitStr>,
    idempotent: Option<syn::LitBool>,
    effect: Option<(MacroToolEffect, proc_macro2::Span)>,
    provenance: Option<(helpers::MacroToolProvenance, proc_macro2::Span)>,
    sanitization: Option<(MacroOutputSanitization, proc_macro2::Span)>,
    has_explicit_sanitization: bool,
    inline_params: Vec<(syn::Ident, syn::LitStr)>,
    env_vars: Vec<(syn::Ident, syn::Lit)>,
    context_fn: Option<syn::Path>,
}

impl ToolAttrBuilder {
    fn parse_params_attr(&mut self, input: syn::parse::ParseStream) -> syn::Result<()> {
        let content;
        syn::parenthesized!(content in input);
        while !content.is_empty() {
            let key: syn::Ident = content.parse()?;
            let _: syn::Token![=] = content.parse()?;
            let value: syn::LitStr = content.parse()?;
            self.inline_params.push((key, value));
            if !content.is_empty() {
                let _: syn::Token![,] = content.parse()?;
            }
        }
        Ok(())
    }

    fn parse_env_attr(&mut self, input: syn::parse::ParseStream) -> syn::Result<()> {
        let content;
        syn::parenthesized!(content in input);
        while !content.is_empty() {
            let key: syn::Ident = content.parse()?;
            let _: syn::Token![=] = content.parse()?;
            let value: syn::Lit = content.parse()?;
            match &value {
                syn::Lit::Str(_) | syn::Lit::Int(_) | syn::Lit::Float(_) | syn::Lit::Bool(_) => {}
                other => {
                    return Err(syn::Error::new(
                        other.span(),
                        "env values must be string, integer, float, or bool literals",
                    ));
                }
            }
            self.env_vars.push((key, value));
            if !content.is_empty() {
                let _: syn::Token![,] = content.parse()?;
            }
        }
        Ok(())
    }

    fn parse_single(&mut self, input: syn::parse::ParseStream) -> syn::Result<()> {
        let ident: syn::Ident = input.parse()?;
        let key = ToolAttrKey::try_from(&ident)?;
        self.parse_by_key(&ident, key, input)
    }

    fn parse_lit_str_attr(
        field: &mut Option<syn::LitStr>,
        ident: &syn::Ident,
        key: ToolAttrKey,
        input: syn::parse::ParseStream,
    ) -> syn::Result<()> {
        let _: syn::Token![=] = input.parse()?;
        if field.is_some() {
            return Err(syn::Error::new(
                ident.span(),
                format!("duplicate `{}` attribute", key.as_str()),
            ));
        }
        *field = Some(input.parse::<syn::LitStr>()?);
        Ok(())
    }

    fn parse_context_attr(
        &mut self,
        ident: &syn::Ident,
        key: ToolAttrKey,
        input: syn::parse::ParseStream,
    ) -> syn::Result<()> {
        let _: syn::Token![=] = input.parse()?;
        if self.context_fn.is_some() {
            return Err(syn::Error::new(
                ident.span(),
                format!("duplicate `{}` attribute", key.as_str()),
            ));
        }
        self.context_fn = Some(input.parse::<syn::Path>()?);
        Ok(())
    }

    fn parse_effect_attr(
        &mut self,
        ident: &syn::Ident,
        key: ToolAttrKey,
        input: syn::parse::ParseStream,
    ) -> syn::Result<()> {
        let _: syn::Token![=] = input.parse()?;
        if self.effect.is_some() {
            return Err(syn::Error::new(
                ident.span(),
                format!("duplicate `{}` attribute", key.as_str()),
            ));
        }
        let span = input.span();
        let eff = if input.peek(syn::Ident) {
            let id: syn::Ident = input.parse()?;
            parse_effect_str(id.to_string().as_str(), id.span())?
        } else if input.peek(syn::LitStr) {
            let lit: syn::LitStr = input.parse()?;
            parse_effect_str(lit.value().as_str(), lit.span())?
        } else {
            return Err(syn::Error::new(
                span,
                "expected identifier or string literal (`read_only`, `mutating`, or `destructive`)",
            ));
        };
        self.effect = Some((eff, ident.span()));
        Ok(())
    }

    fn parse_idempotent_attr(
        &mut self,
        ident: &syn::Ident,
        key: ToolAttrKey,
        input: syn::parse::ParseStream,
    ) -> syn::Result<()> {
        if self.idempotent.is_some() {
            return Err(syn::Error::new(
                ident.span(),
                format!("duplicate `{}` attribute", key.as_str()),
            ));
        }
        if input.peek(syn::Token![=]) {
            let _: syn::Token![=] = input.parse()?;
            self.idempotent = Some(input.parse::<syn::LitBool>()?);
        } else {
            self.idempotent = Some(syn::LitBool::new(true, ident.span()));
        }
        Ok(())
    }

    fn parse_sanitize_parens(
        ident: &syn::Ident,
        input: syn::parse::ParseStream,
    ) -> syn::Result<MacroOutputSanitization> {
        let content;
        syn::parenthesized!(content in input);
        let mut tag: Option<syn::LitStr> = None;
        let mut notice: Option<syn::LitStr> = None;
        while !content.is_empty() {
            let k: syn::Ident = content.parse()?;
            let _: syn::Token![=] = content.parse()?;
            match k.to_string().as_str() {
                ATTR_SANITIZE_TAG_ARG => {
                    if tag.is_some() {
                        return Err(syn::Error::new(k.span(), "duplicate `tag` argument"));
                    }
                    let lit = content.parse::<syn::LitStr>()?;
                    validate_sanitize_tag_lit(&lit)?;
                    tag = Some(lit);
                }
                ATTR_SANITIZE_NOTICE_ARG => {
                    if notice.is_some() {
                        return Err(syn::Error::new(k.span(), "duplicate `notice` argument"));
                    }
                    let lit = content.parse::<syn::LitStr>()?;
                    validate_sanitize_notice_lit(&lit)?;
                    notice = Some(lit);
                }
                _ => {
                    return Err(syn::Error::new(
                        k.span(),
                        "expected `tag` or `notice` in `sanitize(...)`",
                    ));
                }
            }
            if !content.is_empty() {
                let _: syn::Token![,] = content.parse()?;
            }
        }
        let Some(tag_lit) = tag else {
            return Err(syn::Error::new(
                ident.span(),
                "`sanitize(...)` requires a `tag = \"...\"` parameter",
            ));
        };
        if let Some(notice_lit) = notice {
            Ok(MacroOutputSanitization::SanitizeWithNotice {
                tag: tag_lit,
                notice: notice_lit,
            })
        } else {
            Ok(MacroOutputSanitization::SanitizeTagged(tag_lit))
        }
    }

    fn parse_sanitize_attr(
        &mut self,
        ident: &syn::Ident,
        key: ToolAttrKey,
        input: syn::parse::ParseStream,
    ) -> syn::Result<()> {
        if self.has_explicit_sanitization {
            return Err(syn::Error::new(
                ident.span(),
                "`sanitization` cannot be combined with `sanitize` or `quarantine`",
            ));
        }
        if self.sanitization.is_some() {
            return Err(syn::Error::new(
                ident.span(),
                format!("duplicate `{}` attribute", key.as_str()),
            ));
        }
        if input.peek(syn::Token![=]) {
            let _: syn::Token![=] = input.parse()?;
            if input.peek(syn::LitBool) {
                let lit = input.parse::<syn::LitBool>()?;
                let mode = if lit.value() {
                    MacroOutputSanitization::SanitizeTokens
                } else {
                    MacroOutputSanitization::None
                };
                self.sanitization = Some((mode, ident.span()));
            } else if input.peek(syn::LitStr) {
                let lit = input.parse::<syn::LitStr>()?;
                let val = lit.value();
                let mode = match val.as_str() {
                    "none" | "None" => MacroOutputSanitization::None,
                    "sanitize_tokens" | "sanitize" | "SanitizeTokens" => {
                        MacroOutputSanitization::SanitizeTokens
                    }
                    "quarantine" | "Quarantine" => MacroOutputSanitization::Quarantine,
                    _ => {
                        validate_sanitize_tag_lit(&lit)?;
                        MacroOutputSanitization::SanitizeTagged(lit)
                    }
                };
                self.sanitization = Some((mode, ident.span()));
            } else if input.peek(syn::Ident) {
                let id: syn::Ident = input.parse()?;
                let mode = parse_sanitization_str(&id.to_string(), id.span())?;
                self.sanitization = Some((mode, ident.span()));
            } else {
                return Err(syn::Error::new(
                    ident.span(),
                    "expected boolean literal or tag string (e.g. `sanitize = true` or `sanitize = \"tag\"`)",
                ));
            }
        } else if input.peek(syn::token::Paren) {
            let mode = Self::parse_sanitize_parens(ident, input)?;
            self.sanitization = Some((mode, ident.span()));
        } else {
            self.sanitization = Some((MacroOutputSanitization::SanitizeTokens, ident.span()));
        }
        Ok(())
    }

    fn parse_quarantine_attr(
        &mut self,
        ident: &syn::Ident,
        key: ToolAttrKey,
        input: syn::parse::ParseStream,
    ) -> syn::Result<()> {
        if self.has_explicit_sanitization {
            return Err(syn::Error::new(
                ident.span(),
                "`sanitization` cannot be combined with `sanitize` or `quarantine`",
            ));
        }
        if self.sanitization.is_some() {
            return Err(syn::Error::new(
                ident.span(),
                format!("duplicate `{}` attribute", key.as_str()),
            ));
        }
        if input.peek(syn::Token![=]) {
            let _: syn::Token![=] = input.parse()?;
            if input.peek(syn::LitBool) {
                let lit = input.parse::<syn::LitBool>()?;
                let mode = if lit.value() {
                    MacroOutputSanitization::Quarantine
                } else {
                    MacroOutputSanitization::None
                };
                self.sanitization = Some((mode, ident.span()));
            } else if input.peek(syn::LitStr) {
                let lit = input.parse::<syn::LitStr>()?;
                let val = lit.value();
                let mode = if val == "quarantine" || val == "Quarantine" {
                    MacroOutputSanitization::Quarantine
                } else {
                    validate_sanitize_tag_lit(&lit)?;
                    MacroOutputSanitization::SanitizeTagged(lit)
                };
                self.sanitization = Some((mode, ident.span()));
            } else {
                return Err(syn::Error::new(
                    ident.span(),
                    "expected boolean literal or tag string (e.g. `quarantine = true` or `quarantine = \"tag\"`)",
                ));
            }
        } else {
            self.sanitization = Some((MacroOutputSanitization::Quarantine, ident.span()));
        }
        Ok(())
    }

    fn parse_sanitization_attr(
        &mut self,
        ident: &syn::Ident,
        key: ToolAttrKey,
        input: syn::parse::ParseStream,
    ) -> syn::Result<()> {
        let _: syn::Token![=] = input.parse()?;
        if self.has_explicit_sanitization || self.sanitization.is_some() {
            return Err(syn::Error::new(
                ident.span(),
                format!("duplicate `{}` attribute", key.as_str()),
            ));
        }
        let span = input.span();
        let mode = if input.peek(syn::Ident) {
            let id: syn::Ident = input.parse()?;
            parse_sanitization_str(id.to_string().as_str(), id.span())?
        } else if input.peek(syn::LitStr) {
            let lit: syn::LitStr = input.parse()?;
            parse_sanitization_str(lit.value().as_str(), lit.span())?
        } else {
            return Err(syn::Error::new(
                span,
                "expected identifier or string literal (`none`, `sanitize_tokens`, or `quarantine`)",
            ));
        };
        self.has_explicit_sanitization = true;
        self.sanitization = Some((mode, span));
        Ok(())
    }

    pub fn parse_by_key(
        &mut self,
        ident: &syn::Ident,
        key: ToolAttrKey,
        input: syn::parse::ParseStream,
    ) -> syn::Result<()> {
        match key {
            ToolAttrKey::Name => {
                Self::parse_lit_str_attr(&mut self.name_override, ident, key, input)?;
                if let Some(ref lit) = self.name_override
                    && lit.value().trim().is_empty()
                {
                    return Err(syn::Error::new(lit.span(), "`name` must not be empty"));
                }
            }
            ToolAttrKey::Description => {
                Self::parse_lit_str_attr(&mut self.description_inline, ident, key, input)?;
            }
            ToolAttrKey::DescriptionFile => {
                Self::parse_lit_str_attr(&mut self.description_file_path, ident, key, input)?;
            }
            ToolAttrKey::ResponseFile => {
                Self::parse_lit_str_attr(&mut self.response_file_path, ident, key, input)?;
            }
            ToolAttrKey::Response => {
                Self::parse_lit_str_attr(&mut self.response_inline, ident, key, input)?;
            }
            ToolAttrKey::Params => {
                self.parse_params_attr(input)?;
            }
            ToolAttrKey::Env => {
                self.parse_env_attr(input)?;
            }
            ToolAttrKey::Context => {
                self.parse_context_attr(ident, key, input)?;
            }
            ToolAttrKey::Idempotent => {
                self.parse_idempotent_attr(ident, key, input)?;
            }
            ToolAttrKey::Effect => {
                self.parse_effect_attr(ident, key, input)?;
            }
            ToolAttrKey::Provenance => {
                let _: syn::Token![=] = input.parse()?;
                if self.provenance.is_some() {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("duplicate `{}` attribute", key.as_str()),
                    ));
                }
                let span = input.span();
                let prov = if input.peek(syn::Ident) {
                    let id: syn::Ident = input.parse()?;
                    helpers::parse_provenance_str(id.to_string().as_str(), id.span())?
                } else if input.peek(syn::LitStr) {
                    let lit: syn::LitStr = input.parse()?;
                    helpers::parse_provenance_str(lit.value().as_str(), lit.span())?
                } else {
                    return Err(syn::Error::new(
                        span,
                        "expected identifier or string literal (`trusted`, `workspace_local`, `private_data`, `external_mcp`, or `untrusted_network`)",
                    ));
                };
                self.provenance = Some((prov, span));
            }
            ToolAttrKey::Sanitize => {
                self.parse_sanitize_attr(ident, key, input)?;
            }
            ToolAttrKey::Quarantine => {
                self.parse_quarantine_attr(ident, key, input)?;
            }
            ToolAttrKey::Sanitization => {
                self.parse_sanitization_attr(ident, key, input)?;
            }
        }
        Ok(())
    }

    pub fn build(self) -> syn::Result<ToolAttr> {
        validate_tool_attr(&self)?;

        let explicit_effect_span = self
            .effect
            .map(|(_, span)| span)
            .or_else(|| self.idempotent.as_ref().map(syn::LitBool::span));

        let effect = match (self.effect, self.idempotent) {
            (Some((eff, _)), _) => eff,
            (None, Some(idemp)) => {
                if idemp.value() {
                    MacroToolEffect::ReadOnly
                } else {
                    MacroToolEffect::Mutating
                }
            }
            (None, None) => MacroToolEffect::Mutating,
        };
        let idempotent = matches!(effect, MacroToolEffect::ReadOnly);
        let provenance = self
            .provenance
            .map_or_else(helpers::MacroToolProvenance::default, |(p, _)| p);

        let sanitization = self.sanitization.map_or_else(
            || {
                if provenance.is_untrusted_source() {
                    MacroOutputSanitization::Quarantine
                } else {
                    MacroOutputSanitization::default()
                }
            },
            |(mode, _)| match mode {
                MacroOutputSanitization::SanitizeTokens if provenance.is_untrusted_source() => {
                    MacroOutputSanitization::Quarantine
                }
                other => other,
            },
        );

        Ok(ToolAttr {
            name_override: self.name_override,
            description_inline: self.description_inline,
            description_file_path: self.description_file_path,
            response_file_path: self.response_file_path,
            response_inline: self.response_inline,
            inline_params: self.inline_params,
            env_vars: self.env_vars,
            context_fn: self.context_fn,
            idempotent,
            effect,
            explicit_effect_span,
            provenance,
            sanitization,
        })
    }
}

impl syn::parse::Parse for ToolAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut builder = ToolAttrBuilder::default();

        while !input.is_empty() {
            builder.parse_single(input)?;
            if !input.is_empty() {
                let _: syn::Token![,] = input.parse()?;
            }
        }

        builder.build()
    }
}

fn validate_tool_attr(builder: &ToolAttrBuilder) -> syn::Result<()> {
    if builder.description_inline.is_some() && builder.description_file_path.is_some() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`description` and `description_file` are mutually exclusive",
        ));
    }

    if builder.response_file_path.is_some() && builder.response_inline.is_some() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`response` and `response_file` are mutually exclusive",
        ));
    }

    let has_inline_params = !builder.inline_params.is_empty();
    let has_context_fn = builder.context_fn.is_some();
    let has_env = !builder.env_vars.is_empty();

    if has_inline_params && has_context_fn {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`params(...)` and `context = ...` are mutually exclusive; \
             use `params` for compile-time values or `context` for runtime values",
        ));
    }

    if has_inline_params
        && builder.description_file_path.is_none()
        && builder.description_inline.is_none()
    {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`params(...)` requires `description_file = \"...\"` or `description = \"...\"`",
        ));
    }

    if has_context_fn
        && builder.description_file_path.is_none()
        && builder.description_inline.is_none()
    {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`context = ...` requires `description_file = \"...\"` or `description = \"...\"`",
        ));
    }

    if has_env && builder.description_file_path.is_none() && builder.description_inline.is_none() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`env(...)` requires `description_file = \"...\"` or `description = \"...\"`",
        ));
    }

    if let (Some(idemp), Some((eff, span))) = (&builder.idempotent, &builder.effect) {
        let idemp_val = idemp.value();
        let eff_is_read_only = matches!(eff, MacroToolEffect::ReadOnly);
        if idemp_val != eff_is_read_only {
            return Err(syn::Error::new(
                *span,
                "conflicting `idempotent` and `effect` attributes: `idempotent = true` requires `effect = read_only`",
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use syn::parse_str;

    use super::*;

    #[test]
    fn xml_ncname_and_sanitize_attr_validation() {
        assert!(is_valid_xml_ncname("user_query"));
        assert!(is_valid_xml_ncname("_tag-1.2"));
        assert!(!is_valid_xml_ncname(""));
        assert!(!is_valid_xml_ncname("1bad"));
        assert!(!is_valid_xml_ncname("-bad"));
        assert!(!is_valid_xml_ncname("bad:colon"));
        assert!(!is_valid_xml_ncname("bad space"));
        assert!(!is_valid_xml_ncname("bad>tag"));

        assert!(parse_str::<ToolAttr>(r#"sanitize = "valid_tag""#).is_ok());
        assert!(parse_str::<ToolAttr>(r#"sanitize = """#).is_err());
        assert!(parse_str::<ToolAttr>(r#"sanitize = "bad tag""#).is_err());
        assert!(parse_str::<ToolAttr>(r#"sanitize = "surrogate""#).is_err());
        assert!(parse_str::<ToolAttr>(r#"sanitize = "SURROGATE""#).is_err());
        assert!(parse_str::<ToolAttr>(r#"sanitize(tag = "1invalid")"#).is_err());
        assert!(parse_str::<ToolAttr>(r#"quarantine = "bad>tag""#).is_err());

        let oversized_tag = "a".repeat(MAX_SANITIZE_TAG_LEN + 1);
        assert!(parse_str::<ToolAttr>(&format!(r#"sanitize = "{oversized_tag}""#)).is_err());

        let oversized_notice = "n".repeat(MAX_SANITIZE_NOTICE_LEN + 1);
        let notice_input = format!(r#"sanitize(tag = "doc", notice = "{oversized_notice}")"#);
        assert!(parse_str::<ToolAttr>(&notice_input).is_err());
    }
}
