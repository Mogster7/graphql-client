#![deny(missing_docs)]
#![warn(rust_2018_idioms)]
#![allow(clippy::option_option)]

//! Crate for internal use by other graphql-client crates, for code generation.
//!
//! It is not meant to be used directly by users of the library.

use lazy_static::*;
use proc_macro2::TokenStream;
use quote::*;

mod codegen;
mod codegen_options;
/// Deprecation-related code
pub mod deprecation;
/// Contains the [Schema] type and its implementation.
pub mod schema;

mod constants;
mod generated_module;
/// Normalization-related code
pub mod normalization;
mod query;
mod type_qualifiers;

#[cfg(test)]
mod tests;

pub use crate::codegen_options::{CodegenMode, GraphQLClientCodegenOptions};

use std::{collections::HashMap, fmt::Display, io};

#[derive(Debug)]
struct GeneralError(String);

impl Display for GeneralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GeneralError {}

type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;
type CacheMap<T> = std::sync::Mutex<HashMap<std::path::PathBuf, T>>;

lazy_static! {
    static ref SCHEMA_CACHE: CacheMap<schema::Schema> = CacheMap::default();
    static ref QUERY_CACHE: CacheMap<(String, graphql_parser::query::Document<'static, String>)> =
        CacheMap::default();
}

fn get_query(
    path: std::path::PathBuf
) -> Result<(String, graphql_parser::query::Document<'static, String>), BoxError> {
    // Get base relative path from the query file path
    let mut base_path = path.clone();
    assert!(base_path.pop(), "Failed to pop the query path's last element, which suggests it is empty. Should not happen.");

    let lock = QUERY_CACHE.lock().expect("query cache is poisoned");

    // Return an existing entry, if found
    if lock.contains_key(&path) {
        let entry = lock.get(&path).unwrap().clone();
        return Ok(entry);
    }

    // Drop mutex to allow recursive entries to access the query cache
    drop(lock);

    let query_string = read_file(&path)?;

    let mut query = graphql_parser::parse_query(&query_string)
        .map_err(|err| GeneralError(format!("Query parser error: {}", err)))?
        .into_static();

    // Check for imports within the query
    if query_string.find("#import").is_some() {
        let import_split = query_string
            .split("#import")
            // Skip first as it will be empty even if #import is the first string in the file
            .skip(1)
            .collect::<std::vec::Vec<&str>>();

        let imported_paths = import_split.iter().map(|import_path| {
            // Strip quotation marks, trim and get only the filepath
            let stripped_path = import_path.chars()
                .take_while(|&c| c != '\n')
                .collect::<String>()
                .trim()
                .replace("\"", "");

            let relative_path = stripped_path.chars().nth(0) == Some('.');
            if !relative_path {
                panic!("Absolute paths not supported for import statements yet");
            }

            let mut path = base_path.clone();
            path.push(std::path::PathBuf::from(stripped_path));
            path
        }).collect::<Vec<std::path::PathBuf>>();

        for imported_path in imported_paths {
            use graphql_parser::query::Definition;
            let (_, imported_query_document) = get_query(imported_path)?;

            let mut imported_fragments = imported_query_document.definitions.iter().filter(
                |directive| match directive {
                    Definition::Operation(_) => false,
                    Definition::Fragment(_) => true
                }
            ).cloned().collect::<Vec<Definition<'static, String>>>();
            query.definitions.append(&mut imported_fragments);
        }
    }

    let mut lock = QUERY_CACHE.lock().expect("query cache is poisoned");
    let result = (query_string, query);
    lock.insert(path, result.clone());

    Ok(result)
}

/// Generates Rust code given a query document, a schema and options.
pub fn generate_module_token_stream(
    query_path: std::path::PathBuf,
    schema_path: &std::path::Path,
    options: GraphQLClientCodegenOptions,
) -> Result<TokenStream, BoxError> {
    use std::collections::hash_map;

    let schema_extension = schema_path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("INVALID");
    let schema_string;

    // Check the schema cache.
    let schema: schema::Schema = {
        let mut lock = SCHEMA_CACHE.lock().expect("schema cache is poisoned");
        match lock.entry(schema_path.to_path_buf()) {
            hash_map::Entry::Occupied(o) => o.get().clone(),
            hash_map::Entry::Vacant(v) => {
                schema_string = read_file(v.key())?;
                let schema = match schema_extension {
                    "graphql" | "gql" => {
                        let s = graphql_parser::schema::parse_schema::<&str>(&schema_string).map_err(|parser_error| GeneralError(format!("Parser error: {}", parser_error)))?;
                        schema::Schema::from(s)
                    }
                    "json" => {
                        let parsed: graphql_introspection_query::introspection_response::IntrospectionResponse = serde_json::from_str(&schema_string)?;
                        schema::Schema::from(parsed)
                    }
                    extension => return Err(GeneralError(format!("Unsupported extension for the GraphQL schema: {} (only .json and .graphql are supported)", extension)).into())
                };

                v.insert(schema).clone()
            }
        }
    };

    // We need to qualify the query with the path to the crate it is part of
    let (query_string, query) = get_query(
        query_path
    )?;

    let query = crate::query::resolve(&schema, &query)?;

    // Determine which operation we are generating code for. This will be used in operationName.
    let operations = options
        .operation_name
        .as_ref()
        .and_then(|operation_name| query.select_operation(operation_name, *options.normalization()))
        .map(|op| vec![op]);

    let operations = match (operations, &options.mode) {
        (Some(ops), _) => ops,
        (None, &CodegenMode::Cli) => query.operations().collect(),
        (None, &CodegenMode::Derive) => {
            return Err(GeneralError(derive_operation_not_found_error(
                options.struct_ident(),
                &query,
            ))
            .into());
        }
    };

    // The generated modules.
    let mut modules = Vec::with_capacity(operations.len());

    for operation in &operations {
        let generated = generated_module::GeneratedModule {
            query_string: query_string.as_str(),
            schema: &schema,
            resolved_query: &query,
            operation: &operation.1.name,
            options: &options,
        }
        .to_token_stream()?;
        modules.push(generated);
    }

    let modules = quote! { #(#modules)* };

    Ok(modules)
}

#[derive(Debug)]
enum ReadFileError {
    FileNotFound { path: String, io_error: io::Error },
    ReadError { path: String, io_error: io::Error },
}

impl Display for ReadFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadFileError::FileNotFound { path, .. } => {
                write!(f, "Could not find file with path: {}\n
                Hint: file paths in the GraphQLQuery attribute are relative to the project root (location of the Cargo.toml). Example: query_path = \"src/my_query.graphql\".", path)
            }
            ReadFileError::ReadError { path, .. } => {
                f.write_str("Error reading file at: ")?;
                f.write_str(path)
            }
        }
    }
}

impl std::error::Error for ReadFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ReadFileError::FileNotFound { io_error, .. }
            | ReadFileError::ReadError { io_error, .. } => Some(io_error),
        }
    }
}

fn read_file(path: &std::path::Path) -> Result<String, ReadFileError> {
    use std::fs;
    use std::io::prelude::*;

    let mut out = String::new();
    let mut file = fs::File::open(path).map_err(|io_error| ReadFileError::FileNotFound {
        io_error,
        path: path.display().to_string(),
    })?;

    file.read_to_string(&mut out)
        .map_err(|io_error| ReadFileError::ReadError {
            io_error,
            path: path.display().to_string(),
        })?;
    Ok(out)
}

/// In derive mode, build an error when the operation with the same name as the struct is not found.
fn derive_operation_not_found_error(
    ident: Option<&proc_macro2::Ident>,
    query: &crate::query::Query,
) -> String {
    let operation_name = ident.map(ToString::to_string);
    let struct_ident = operation_name.as_deref().unwrap_or("");

    let available_operations: Vec<&str> = query
        .operations()
        .map(|(_id, op)| op.name.as_str())
        .collect();
    let available_operations: String = available_operations.join(", ");

    return format!(
        "The struct name does not match any defined operation in the query file.\nStruct name: {}\nDefined operations: {}",
        struct_ident,
        available_operations,
    );
}
