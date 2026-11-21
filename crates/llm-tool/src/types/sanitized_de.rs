//! Sanitizing serde [`Deserializer`](serde::Deserializer) adapter for [`Sanitized<T>`](super::Sanitized)
//! and [`Untrusted<T>`](super::Untrusted) tool input parameters.
//!
//! Ensures strings and map keys inside deserialized tool arguments automatically have
//! LLM control tokens, Bidi overrides, invisible smuggling characters, and quarantine
//! breakouts neutralized (`no_std + alloc`, 100% `unsafe`-free).

use alloc::{borrow::Cow, string::String, vec::Vec};
use core::fmt;

use serde::de::{
    DeserializeSeed, Deserializer, EnumAccess, MapAccess, SeqAccess, VariantAccess, Visitor,
};

use super::OutputSanitization;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum InputSanitizeMode {
    SanitizeTokens,
    Quarantine,
}

impl InputSanitizeMode {
    fn apply_value_str(self, s: &str) -> Cow<'_, str> {
        match self {
            Self::SanitizeTokens => OutputSanitization::SanitizeTokens.apply_str(s),
            Self::Quarantine => OutputSanitization::Quarantine.apply_str(s),
        }
    }

    fn apply_key_str(s: &str) -> Cow<'_, str> {
        OutputSanitization::SanitizeTokens.apply_str(s)
    }
}

pub(super) struct SanitizingDeserializer<D> {
    inner: D,
    mode: InputSanitizeMode,
    is_identifier_or_map_key: bool,
}

impl<D> SanitizingDeserializer<D> {
    pub(super) const fn new(inner: D, mode: InputSanitizeMode) -> Self {
        Self {
            inner,
            mode,
            is_identifier_or_map_key: false,
        }
    }

    const fn new_key(inner: D, mode: InputSanitizeMode) -> Self {
        Self {
            inner,
            mode,
            is_identifier_or_map_key: true,
        }
    }
}

struct SanitizingVisitor<V> {
    inner: V,
    mode: InputSanitizeMode,
    is_identifier_or_map_key: bool,
}

impl<V> SanitizingVisitor<V> {
    fn transform<'a>(&self, s: &'a str) -> Cow<'a, str> {
        if self.is_identifier_or_map_key {
            InputSanitizeMode::apply_key_str(s)
        } else {
            self.mode.apply_value_str(s)
        }
    }
}

impl<'de, V: Visitor<'de>> Visitor<'de> for SanitizingVisitor<V> {
    type Value = V::Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.expecting(formatter)
    }

    fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Self::Value, E> {
        self.inner.visit_bool(v)
    }

    fn visit_i8<E: serde::de::Error>(self, v: i8) -> Result<Self::Value, E> {
        self.inner.visit_i8(v)
    }

    fn visit_i16<E: serde::de::Error>(self, v: i16) -> Result<Self::Value, E> {
        self.inner.visit_i16(v)
    }

    fn visit_i32<E: serde::de::Error>(self, v: i32) -> Result<Self::Value, E> {
        self.inner.visit_i32(v)
    }

    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
        self.inner.visit_i64(v)
    }

    fn visit_i128<E: serde::de::Error>(self, v: i128) -> Result<Self::Value, E> {
        self.inner.visit_i128(v)
    }

    fn visit_u8<E: serde::de::Error>(self, v: u8) -> Result<Self::Value, E> {
        self.inner.visit_u8(v)
    }

    fn visit_u16<E: serde::de::Error>(self, v: u16) -> Result<Self::Value, E> {
        self.inner.visit_u16(v)
    }

    fn visit_u32<E: serde::de::Error>(self, v: u32) -> Result<Self::Value, E> {
        self.inner.visit_u32(v)
    }

    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
        self.inner.visit_u64(v)
    }

    fn visit_u128<E: serde::de::Error>(self, v: u128) -> Result<Self::Value, E> {
        self.inner.visit_u128(v)
    }

    fn visit_f32<E: serde::de::Error>(self, v: f32) -> Result<Self::Value, E> {
        self.inner.visit_f32(v)
    }

    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
        self.inner.visit_f64(v)
    }

    fn visit_char<E: serde::de::Error>(self, v: char) -> Result<Self::Value, E> {
        let mut buf = [0u8; 4];
        let s = v.encode_utf8(&mut buf);
        match self.transform(s) {
            Cow::Borrowed(_) => self.inner.visit_char(v),
            Cow::Owned(clean) => self.inner.visit_string(clean),
        }
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
        match self.transform(v) {
            Cow::Borrowed(b) => self.inner.visit_str(b),
            Cow::Owned(clean) => self.inner.visit_string(clean),
        }
    }

    fn visit_borrowed_str<E: serde::de::Error>(self, v: &'de str) -> Result<Self::Value, E> {
        match self.transform(v) {
            Cow::Borrowed(_) => self.inner.visit_borrowed_str(v),
            Cow::Owned(clean) => self.inner.visit_string(clean),
        }
    }

    fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
        match self.transform(&v) {
            Cow::Borrowed(_) => self.inner.visit_string(v),
            Cow::Owned(clean) => self.inner.visit_string(clean),
        }
    }

    fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
        self.inner.visit_bytes(v)
    }

    fn visit_borrowed_bytes<E: serde::de::Error>(self, v: &'de [u8]) -> Result<Self::Value, E> {
        self.inner.visit_borrowed_bytes(v)
    }

    fn visit_byte_buf<E: serde::de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
        self.inner.visit_byte_buf(v)
    }

    fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        self.inner.visit_none()
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        self.inner
            .visit_some(SanitizingDeserializer::new(deserializer, self.mode))
    }

    fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        self.inner.visit_unit()
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        self.inner
            .visit_newtype_struct(SanitizingDeserializer::new(deserializer, self.mode))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
        self.inner.visit_seq(SanitizingSeqAccess {
            inner: seq,
            mode: self.mode,
        })
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
        self.inner.visit_map(SanitizingMapAccess {
            inner: map,
            mode: self.mode,
        })
    }

    fn visit_enum<A: EnumAccess<'de>>(self, data: A) -> Result<Self::Value, A::Error> {
        self.inner.visit_enum(SanitizingEnumAccess {
            inner: data,
            mode: self.mode,
        })
    }
}

struct SanitizingSeed<S> {
    inner: S,
    mode: InputSanitizeMode,
    is_key: bool,
}

impl<'de, S: DeserializeSeed<'de>> DeserializeSeed<'de> for SanitizingSeed<S> {
    type Value = S::Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        let wrapped = if self.is_key {
            SanitizingDeserializer::new_key(deserializer, self.mode)
        } else {
            SanitizingDeserializer::new(deserializer, self.mode)
        };
        self.inner.deserialize(wrapped)
    }
}

struct SanitizingSeqAccess<A> {
    inner: A,
    mode: InputSanitizeMode,
}

impl<'de, A: SeqAccess<'de>> SeqAccess<'de> for SanitizingSeqAccess<A> {
    type Error = A::Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, Self::Error> {
        self.inner.next_element_seed(SanitizingSeed {
            inner: seed,
            mode: self.mode,
            is_key: false,
        })
    }

    fn size_hint(&self) -> Option<usize> {
        self.inner.size_hint()
    }
}

struct SanitizingMapAccess<A> {
    inner: A,
    mode: InputSanitizeMode,
}

impl<'de, A: MapAccess<'de>> MapAccess<'de> for SanitizingMapAccess<A> {
    type Error = A::Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, Self::Error> {
        self.inner.next_key_seed(SanitizingSeed {
            inner: seed,
            mode: self.mode,
            is_key: true,
        })
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(
        &mut self,
        seed: V,
    ) -> Result<V::Value, Self::Error> {
        self.inner.next_value_seed(SanitizingSeed {
            inner: seed,
            mode: self.mode,
            is_key: false,
        })
    }

    fn size_hint(&self) -> Option<usize> {
        self.inner.size_hint()
    }
}

struct SanitizingEnumAccess<A> {
    inner: A,
    mode: InputSanitizeMode,
}

impl<'de, A: EnumAccess<'de>> EnumAccess<'de> for SanitizingEnumAccess<A> {
    type Error = A::Error;
    type Variant = SanitizingVariantAccess<A::Variant>;

    fn variant_seed<V: DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Self::Variant), Self::Error> {
        let (val, var) = self.inner.variant_seed(SanitizingSeed {
            inner: seed,
            mode: self.mode,
            is_key: true,
        })?;
        Ok((
            val,
            SanitizingVariantAccess {
                inner: var,
                mode: self.mode,
            },
        ))
    }
}

struct SanitizingVariantAccess<A> {
    inner: A,
    mode: InputSanitizeMode,
}

impl<'de, A: VariantAccess<'de>> VariantAccess<'de> for SanitizingVariantAccess<A> {
    type Error = A::Error;

    fn unit_variant(self) -> Result<(), Self::Error> {
        self.inner.unit_variant()
    }

    fn newtype_variant_seed<T: DeserializeSeed<'de>>(
        self,
        seed: T,
    ) -> Result<T::Value, Self::Error> {
        self.inner.newtype_variant_seed(SanitizingSeed {
            inner: seed,
            mode: self.mode,
            is_key: false,
        })
    }

    fn tuple_variant<V: Visitor<'de>>(
        self,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.inner.tuple_variant(
            len,
            SanitizingVisitor {
                inner: visitor,
                mode: self.mode,
                is_identifier_or_map_key: false,
            },
        )
    }

    fn struct_variant<V: Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.inner.struct_variant(
            fields,
            SanitizingVisitor {
                inner: visitor,
                mode: self.mode,
                is_identifier_or_map_key: false,
            },
        )
    }
}

macro_rules! delegate_deserialize {
    ($method:ident) => {
        fn $method<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
            self.inner.$method(SanitizingVisitor {
                inner: visitor,
                mode: self.mode,
                is_identifier_or_map_key: self.is_identifier_or_map_key,
            })
        }
    };
}

impl<'de, D: Deserializer<'de>> Deserializer<'de> for SanitizingDeserializer<D> {
    type Error = D::Error;

    delegate_deserialize!(deserialize_any);
    delegate_deserialize!(deserialize_bool);
    delegate_deserialize!(deserialize_i8);
    delegate_deserialize!(deserialize_i16);
    delegate_deserialize!(deserialize_i32);
    delegate_deserialize!(deserialize_i64);
    delegate_deserialize!(deserialize_i128);
    delegate_deserialize!(deserialize_u8);
    delegate_deserialize!(deserialize_u16);
    delegate_deserialize!(deserialize_u32);
    delegate_deserialize!(deserialize_u64);
    delegate_deserialize!(deserialize_u128);
    delegate_deserialize!(deserialize_f32);
    delegate_deserialize!(deserialize_f64);
    delegate_deserialize!(deserialize_char);
    delegate_deserialize!(deserialize_str);
    delegate_deserialize!(deserialize_string);
    delegate_deserialize!(deserialize_bytes);
    delegate_deserialize!(deserialize_byte_buf);
    delegate_deserialize!(deserialize_option);
    delegate_deserialize!(deserialize_unit);
    delegate_deserialize!(deserialize_seq);
    delegate_deserialize!(deserialize_map);
    delegate_deserialize!(deserialize_ignored_any);

    fn deserialize_identifier<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        self.inner.deserialize_identifier(SanitizingVisitor {
            inner: visitor,
            mode: self.mode,
            is_identifier_or_map_key: true,
        })
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.inner.deserialize_unit_struct(
            name,
            SanitizingVisitor {
                inner: visitor,
                mode: self.mode,
                is_identifier_or_map_key: self.is_identifier_or_map_key,
            },
        )
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.inner.deserialize_newtype_struct(
            name,
            SanitizingVisitor {
                inner: visitor,
                mode: self.mode,
                is_identifier_or_map_key: self.is_identifier_or_map_key,
            },
        )
    }

    fn deserialize_tuple<V: Visitor<'de>>(
        self,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.inner.deserialize_tuple(
            len,
            SanitizingVisitor {
                inner: visitor,
                mode: self.mode,
                is_identifier_or_map_key: self.is_identifier_or_map_key,
            },
        )
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.inner.deserialize_tuple_struct(
            name,
            len,
            SanitizingVisitor {
                inner: visitor,
                mode: self.mode,
                is_identifier_or_map_key: self.is_identifier_or_map_key,
            },
        )
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.inner.deserialize_struct(
            name,
            fields,
            SanitizingVisitor {
                inner: visitor,
                mode: self.mode,
                is_identifier_or_map_key: self.is_identifier_or_map_key,
            },
        )
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.inner.deserialize_enum(
            name,
            variants,
            SanitizingVisitor {
                inner: visitor,
                mode: self.mode,
                is_identifier_or_map_key: self.is_identifier_or_map_key,
            },
        )
    }

    fn is_human_readable(&self) -> bool {
        self.inner.is_human_readable()
    }
}
