use field_type::FieldType;
use objects::GqlObjectField;
use proc_macro2::{Ident, Span};

pub const TYPENAME_FIELD: &str = "__typename";

pub fn string_type() -> Ident {
    Ident::new("String", Span::call_site())
}

#[cfg(test)]
pub fn float_type() -> Ident {
    Ident::new("Float", Span::call_site())
}

pub fn typename_field() -> GqlObjectField {
    GqlObjectField {
        description: None,
        name: TYPENAME_FIELD.to_string(),
        /// Non-nullable, see spec:
        /// https://github.com/facebook/graphql/blob/master/spec/Section%204%20--%20Introspection.md
        type_: FieldType::Named(string_type()),
    }
}

pub const MULTIPLE_SUBSCRIPTION_FIELDS_ERROR: &str = r##"
Multiple-field queries on the root subscription field are forbidden by the spec.

See: https://github.com/facebook/graphql/blob/master/spec/Section%205%20--%20Validation.md#subscription-operation-definitions
"##;
