//! Compile-time type-state builder derive for `md-tmpl`-generated structs.
//!
//! `md-tmpl-macros` derives `#cp::__private::TypedBuilder` on generated template
//! parameter and response structs. Unlike `typed-builder`, which emits uninhabited
//! `enum {Struct}_Error_Missing_required_field_{field} {}` items that trigger
//! `clippy::empty_enums` under `#![deny(clippy::nursery)]`, this derive encodes
//! field presence purely via tuple type-state (`()` vs `(T,)`), emitting zero
//! empty enums and requiring zero lint suppressions.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Field, Fields, Ident, Type};

const ATTR_BUILDER: &str = "builder";
const KEY_CRATE_MODULE_PATH: &str = "crate_module_path";
const KEY_DEFAULT: &str = "default";
const KEY_SETTER: &str = "setter";
const KEY_INTO: &str = "into";

/// Parsed `#[builder(...)]` options for a single struct field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FieldBuilderOpts {
    has_default: bool,
    setter_into: bool,
}

struct ParsedField<'a> {
    ident: &'a Ident,
    ty: &'a Type,
    opts: FieldBuilderOpts,
}

/// Entry point for `#[derive(ResponseStructBuilder)]`.
pub fn derive_impl(input: &DeriveInput) -> syn::Result<TokenStream> {
    validate_struct_builder_attrs(&input.attrs)?;
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "ResponseStructBuilder does not support generic structs",
        ));
    }

    let Data::Struct(data_struct) = &input.data else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "ResponseStructBuilder only supports named structs",
        ));
    };
    let Fields::Named(named) = &data_struct.fields else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "ResponseStructBuilder requires named fields",
        ));
    };

    let mut parsed_fields = Vec::with_capacity(named.named.len());
    for field in &named.named {
        parsed_fields.push(parse_field(field)?);
    }

    let struct_ident = &input.ident;
    let vis = &input.vis;
    let builder_ident = format_ident!("{}Builder", struct_ident);

    if parsed_fields.is_empty() {
        return Ok(emit_empty_struct_builder(struct_ident, &builder_ident, vis));
    }

    Ok(emit_tuple_state_builder(
        struct_ident,
        &builder_ident,
        vis,
        &parsed_fields,
    ))
}

fn validate_struct_builder_attrs(attrs: &[syn::Attribute]) -> syn::Result<()> {
    for attr in attrs {
        if !attr.path().is_ident(ATTR_BUILDER) {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident(KEY_CRATE_MODULE_PATH) {
                let _eq: syn::Token![=] = meta.input.parse()?;
                let _path: syn::Path = meta.input.parse()?;
                Ok(())
            } else {
                Err(meta.error("unsupported struct-level #[builder(...)] option"))
            }
        })?;
    }
    Ok(())
}

fn parse_field(field: &Field) -> syn::Result<ParsedField<'_>> {
    let ident = field
        .ident
        .as_ref()
        .ok_or_else(|| syn::Error::new_spanned(field, "expected named field"))?;
    let mut opts = FieldBuilderOpts::default();
    for attr in &field.attrs {
        if !attr.path().is_ident(ATTR_BUILDER) {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident(KEY_DEFAULT) {
                opts.has_default = true;
                Ok(())
            } else if meta.path.is_ident(KEY_SETTER) {
                meta.parse_nested_meta(|nested| {
                    if nested.path.is_ident(KEY_INTO) {
                        opts.setter_into = true;
                        Ok(())
                    } else {
                        Err(nested.error("unsupported #[builder(setter(...))] option"))
                    }
                })
            } else {
                Err(meta.error("unsupported field-level #[builder(...)] option"))
            }
        })?;
    }
    Ok(ParsedField {
        ident,
        ty: &field.ty,
        opts,
    })
}

fn emit_empty_struct_builder(
    struct_ident: &Ident,
    builder_ident: &Ident,
    vis: &syn::Visibility,
) -> TokenStream {
    quote! {
        #[doc(hidden)]
        #vis struct #builder_ident {
            _priv: (),
        }

        impl #struct_ident {
            #[doc(hidden)]
            #[must_use]
            pub const fn builder() -> #builder_ident {
                #builder_ident { _priv: () }
            }
        }

        impl #builder_ident {
            #[doc(hidden)]
            #[must_use]
            pub const fn build(self) -> #struct_ident {
                let Self { _priv: () } = self;
                #struct_ident {}
            }
        }
    }
}

fn emit_tuple_state_builder(
    struct_ident: &Ident,
    builder_ident: &Ident,
    vis: &syn::Visibility,
    fields: &[ParsedField<'_>],
) -> TokenStream {
    let count = fields.len();
    let state_generics: Vec<Ident> = (0..count).map(|i| format_ident!("__State{i}")).collect();
    let slot_vars: Vec<Ident> = (0..count).map(|i| format_ident!("__slot_{i}")).collect();

    let init_slot_types: Vec<TokenStream> = fields
        .iter()
        .map(|f| {
            let ty = f.ty;
            if f.opts.has_default {
                quote! { (#ty,) }
            } else {
                quote! { () }
            }
        })
        .collect();

    let init_slot_exprs: Vec<TokenStream> = fields
        .iter()
        .map(|f| {
            let ty = f.ty;
            if f.opts.has_default {
                quote! { (<#ty as ::core::default::Default>::default(),) }
            } else {
                quote! { () }
            }
        })
        .collect();

    let done_slot_types: Vec<TokenStream> = fields
        .iter()
        .map(|f| {
            let ty = f.ty;
            quote! { (#ty,) }
        })
        .collect();

    let field_idents: Vec<&Ident> = fields.iter().map(|f| f.ident).collect();
    let setter_methods: Vec<TokenStream> = fields
        .iter()
        .enumerate()
        .map(|(idx, field)| {
            emit_setter_method(builder_ident, idx, field, &state_generics, &slot_vars)
        })
        .collect();

    let init_tuple_ty = quote! { (#(#init_slot_types,)*) };
    let done_tuple_ty = quote! { (#(#done_slot_types,)*) };

    quote! {
        #[doc(hidden)]
        #vis struct #builder_ident<__State = #init_tuple_ty> {
            fields: __State,
        }

        #[automatically_derived]
        impl ::core::default::Default for #builder_ident<#init_tuple_ty> {
            fn default() -> Self {
                Self {
                    fields: (#(#init_slot_exprs,)*),
                }
            }
        }

        impl #struct_ident {
            #[doc(hidden)]
            #[must_use]
            pub fn builder() -> #builder_ident<#init_tuple_ty> {
                <#builder_ident<#init_tuple_ty> as ::core::default::Default>::default()
            }
        }

        impl<#(#state_generics),*> #builder_ident<(#(#state_generics,)*)> {
            #(#setter_methods)*
        }

        #[automatically_derived]
        impl ::core::convert::From<#done_tuple_ty> for #struct_ident {
            fn from((#((#slot_vars,),)*): #done_tuple_ty) -> Self {
                Self {
                    #(#field_idents: #slot_vars,)*
                }
            }
        }

        impl #builder_ident<#done_tuple_ty> {
            #[doc(hidden)]
            #[must_use]
            pub fn build(self) -> #struct_ident {
                ::core::convert::From::from(self.fields)
            }
        }
    }
}

fn emit_setter_method(
    builder_ident: &Ident,
    target_idx: usize,
    field: &ParsedField<'_>,
    state_generics: &[Ident],
    slot_vars: &[Ident],
) -> TokenStream {
    let method_ident = field.ident;
    let field_ty = field.ty;
    let replaced_var = &slot_vars[target_idx];

    let out_types: Vec<TokenStream> = state_generics
        .iter()
        .enumerate()
        .map(|(i, g)| {
            if i == target_idx {
                quote! { (#field_ty,) }
            } else {
                quote! { #g }
            }
        })
        .collect();

    let (sig_tokens, val_expr) = if field.opts.setter_into {
        (
            quote! {
                pub fn #method_ident<__IntoValue: ::core::convert::Into<#field_ty>>(
                    self,
                    __value: __IntoValue,
                ) -> #builder_ident<(#(#out_types,)*)>
            },
            quote! { (::core::convert::Into::into(__value),) },
        )
    } else {
        (
            quote! {
                pub fn #method_ident(
                    self,
                    __value: #field_ty,
                ) -> #builder_ident<(#(#out_types,)*)>
            },
            quote! { (__value,) },
        )
    };

    let out_exprs: Vec<TokenStream> = slot_vars
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if i == target_idx {
                val_expr.clone()
            } else {
                quote! { #v }
            }
        })
        .collect();

    quote! {
        #[doc(hidden)]
        #[must_use]
        #sig_tokens {
            let Self {
                fields: (#(#slot_vars,)*),
            } = self;
            ::core::mem::drop(#replaced_var);
            #builder_ident {
                fields: (#(#out_exprs,)*),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use syn::parse_quote;

    use super::*;

    #[test]
    fn generates_builder_without_empty_enums() {
        let input: DeriveInput = parse_quote! {
            #[builder(crate_module_path = ::llm_tool::__md_tmpl::__private::typed_builder)]
            pub struct Weather {
                #[builder(setter(into))]
                pub city: String,
                pub temp_f: i64,
                #[builder(default, setter(into))]
                pub notes: Vec<String>,
            }
        };
        let tokens = derive_impl(&input).expect("derive succeeds").to_string();
        assert!(tokens.contains("WeatherBuilder"));
        assert!(!tokens.contains("enum "));
    }
}
