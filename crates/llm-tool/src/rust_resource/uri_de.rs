//! Type-aware `serde::Deserializer` over extracted URI template variable slices.
//!
//! Supports `u8..u128`, `i8..i128`, `usize`, `isize`, `f32`, `f64`, `bool`, `char`,
//! `Option<T>`, strings, newtypes, and unit enums (`no_std + alloc`, `unsafe`-free).

use alloc::format;

use serde::{
    de::{Deserializer, Error as _, Visitor, value::BorrowedStrDeserializer},
    forward_to_deserialize_any,
};

/// Single-value URI template variable deserializer parsing primitives from a borrowed `&'de str`.
pub(super) struct UriVarDeserializer<'de> {
    raw: &'de str,
}

impl<'de> UriVarDeserializer<'de> {
    pub(super) const fn new(raw: &'de str) -> Self {
        Self { raw }
    }
}

macro_rules! impl_parse_int_or_float {
    ($method:ident, $visit:ident, $ty:ty) => {
        fn $method<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where
            V: Visitor<'de>,
        {
            let parsed = self.raw.parse::<$ty>().map_err(|err| {
                serde::de::value::Error::custom(format!(
                    "invalid {} '{}' in URI variable: {err}",
                    stringify!($ty),
                    self.raw
                ))
            })?;
            visitor.$visit(parsed)
        }
    };
}

impl<'de> Deserializer<'de> for UriVarDeserializer<'de> {
    type Error = serde::de::value::Error;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_borrowed_str(self.raw)
    }

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        match self.raw {
            "true" | "1" => visitor.visit_bool(true),
            "false" | "0" => visitor.visit_bool(false),
            other => Err(serde::de::value::Error::custom(format!(
                "invalid bool '{other}' in URI variable (expected 'true' or 'false')"
            ))),
        }
    }

    impl_parse_int_or_float!(deserialize_i8, visit_i8, i8);
    impl_parse_int_or_float!(deserialize_i16, visit_i16, i16);
    impl_parse_int_or_float!(deserialize_i32, visit_i32, i32);
    impl_parse_int_or_float!(deserialize_i64, visit_i64, i64);
    impl_parse_int_or_float!(deserialize_i128, visit_i128, i128);
    impl_parse_int_or_float!(deserialize_u8, visit_u8, u8);
    impl_parse_int_or_float!(deserialize_u16, visit_u16, u16);
    impl_parse_int_or_float!(deserialize_u32, visit_u32, u32);
    impl_parse_int_or_float!(deserialize_u64, visit_u64, u64);
    impl_parse_int_or_float!(deserialize_u128, visit_u128, u128);
    impl_parse_int_or_float!(deserialize_f32, visit_f32, f32);
    impl_parse_int_or_float!(deserialize_f64, visit_f64, f64);

    fn deserialize_char<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let mut chars = self.raw.chars();
        match (chars.next(), chars.next()) {
            (Some(ch), None) => visitor.visit_char(ch),
            _ => Err(serde::de::value::Error::custom(format!(
                "invalid single char '{}' in URI variable",
                self.raw
            ))),
        }
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_borrowed_str(self.raw)
    }

    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_borrowed_str(self.raw)
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        if self.raw.is_empty() {
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_enum<V>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        BorrowedStrDeserializer::new(self.raw).deserialize_enum(name, variants, visitor)
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_borrowed_str(self.raw)
    }

    forward_to_deserialize_any! {
        bytes byte_buf unit unit_struct seq tuple tuple_struct map struct ignored_any
    }
}

impl<'de> serde::de::IntoDeserializer<'de, serde::de::value::Error> for UriVarDeserializer<'de> {
    type Deserializer = Self;

    fn into_deserializer(self) -> Self::Deserializer {
        self
    }
}
