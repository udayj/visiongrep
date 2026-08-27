# AGENTS.md

# visiongrep — Rust engineering rules for coding agents

Read this file before writing or modifying code.

This file is the **source of truth** for engineering style and architectural expectations in this
repository. Existing code may have been AI-generated and is **not automatically precedent**.

When touching existing code:

- follow this file even when nearby code uses a weaker pattern,
- improve directly touched code toward these rules when the change is small and relevant,
- do not perform broad unrelated rewrites,
- preserve externally observable behavior unless the task intentionally changes it.

Write Rust that is:

- correct,
- simple,
- idiomatic,
- ergonomic,
- strongly typed,
- easy to review,
- unsurprising to an experienced Rust developer.

The target is the style of strong production Rust codebases such as `ripgrep`, `axum`, `clap`,
and `fd`: boring implementation, excellent APIs, deliberate ownership, useful typed errors,
clear module boundaries, and minimal accidental complexity.

**Do not write clever Rust. Write boring Rust with excellent ergonomics.**

When principles conflict, prefer:

1. correctness and safety,
2. simplicity and local reasoning,
3. type safety and API ergonomics,
4. maintainability,
5. compatibility,
6. measured performance,
7. terseness.

---

# 1. General design principles

## Optimize for local reasoning

A reader should be able to understand a function mostly from:

- its name,
- its signature,
- its types,
- and a small amount of nearby code.

Prefer explicit data flow over:

- hidden coupling,
- implicit global state,
- magic ordering,
- distant invariants,
- surprising trait machinery,
- side effects hidden behind innocent-looking APIs.

If understanding a function requires remembering several unrelated facts from distant modules,
the design is probably too coupled.

---

## Make invalid states hard to represent

Prefer types that encode domain rules.

Use:

- enums for closed states,
- newtypes for semantically distinct values,
- `Option<T>` for genuine absence,
- `Result<T, E>` for recoverable failure,
- structs once tuple fields have semantic meaning.

Avoid:

- stringly typed states,
- integer sentinel values,
- ambiguous boolean parameters,
- multiple loosely related `Option` fields that permit impossible combinations,
- maps when the schema is actually known,
- unvalidated primitive values flowing deep into the program.

Prefer:

```rust
enum OutputFormat {
    Text,
    Json,
}
```

over:

```rust
let output_format: String;
```

when the valid values are known.

Use typestate only when it materially improves an important API. Do not turn ordinary application
logic into a type-system puzzle.

---

## Prefer the least powerful construct that solves the problem

Prefer, roughly:

- immutable value over mutable value,
- local ownership over shared ownership,
- concrete type over generic type,
- generic type over trait object,
- function over custom trait,
- ordinary Rust over macro,
- synchronous code over async when asynchronous I/O is not needed.

Use more powerful machinery only when its semantics are actually required.

---

## No speculative abstractions

Do not add:

- a trait for one implementation,
- a generic parameter for one concrete use,
- a builder for two obvious required values,
- a wrapper type with no invariant or semantic value,
- a module for one trivial helper,
- an `Arc` without real shared ownership,
- a lock without real shared mutation,
- async simply because concurrency exists,
- a cache without evidence that it is useful.

Abstract repeated or structurally obvious needs, not imagined future requirements.

---

# 2. Existing code is not policy

The repository may contain earlier AI-generated design decisions.

Do not copy a pattern merely because it already appears in the codebase.

Before reusing an existing pattern, ask whether it satisfies this file.

Appropriate local improvements while implementing a task include:

- removing an unnecessary `clone`,
- replacing string state with an enum,
- introducing a typed error,
- removing unnecessary shared mutable state,
- simplifying nested control flow,
- replacing ambiguous booleans with meaningful types,
- improving a touched API so invalid use is harder.

Do not use a small task as an excuse for a repository-wide cleanup.

---

# 3. Formatting

Use standard `rustfmt`.

Before completion, run:

```text
cargo fmt --all
cargo fmt --all --check
```

Do not manually format Rust in ways that fight `rustfmt`.

Prefer ordinary rustfmt-compatible style:

- trailing commas in multiline constructs,
- one logical item per line when multiline,
- normal indentation,
- no hand-aligned `=`, `=>`, fields, or comments,
- no decorative formatting.

If rustfmt produces code that is still hard to read, simplify the expression or structure rather
than trying to manually format around it.

---

# 4. Lints

Compiler and Clippy warnings are errors.

The default lint command is:

```text
cargo clippy --workspace --all-targets -- -D warnings
```

If all feature combinations are compatible, also use:

```text
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Do not blindly use `--all-features` when features are intentionally incompatible. In that case,
check the supported relevant feature combinations explicitly.

Do not enable these Clippy groups wholesale:

- `clippy::pedantic`,
- `clippy::nursery`,
- `clippy::restriction`.

Individual lints from those groups may be enabled when they encode a useful repository-wide rule.

Do not silence a warning merely to make CI pass.

When `#[allow(...)]` is genuinely justified:

- scope it as narrowly as possible,
- prefer item-level over crate-level allows,
- include a short explanation when the reason is not obvious.

Warnings should normally be fixed at the cause.

---

# 5. Ownership and borrowing

## Borrow by default; own deliberately

Use natural borrowed inputs:

- `&str` instead of `&String`,
- `&Path` instead of `&PathBuf`,
- `&[T]` instead of `&Vec<T>`,
- `&T` instead of cloned `T`.

Use owned values when the callee:

- retains the value,
- transfers it,
- needs independent ownership,
- intentionally snapshots it.

Do not spread lifetimes through an API merely to save an insignificant allocation.

Ownership should reflect semantics, not just compiler convenience.

---

## Do not clone to appease the borrow checker

A `clone()` should have a semantic reason.

Before cloning, consider:

- shortening a borrow,
- reordering operations,
- moving ownership,
- destructuring,
- borrowing one field instead of the whole struct,
- using a helper to narrow a borrow,
- `mem::take`,
- `Option::take`,
- changing the data representation.

Cloning is correct when independent ownership is actually required.

When cloning an `Arc`, prefer:

```rust
Arc::clone(&state)
```

over:

```rust
state.clone()
```

because shared ownership is then explicit.

**Do not use cloning as the default answer to an ownership design problem.**

---

## Avoid gratuitous allocation

Do not allocate a:

- `String`,
- `Vec`,
- `PathBuf`,
- intermediate collection,

when a borrowed value, slice, or iterator is sufficient.

Do not collect an iterator merely to iterate it once.

Use `Cow` only when genuine borrow-or-own semantics improve the API.

Do not create complicated lifetime machinery just to eliminate tiny allocations without evidence
that those allocations matter.

---

# 6. API design

## Let types carry meaning

Prefer:

```rust
fn set_mode(mode: Mode)
```

over:

```rust
fn set_mode(mode: &str)
```

when valid modes are known.

Prefer:

```rust
fn retry(policy: RetryPolicy)
```

over:

```rust
fn retry(enabled: bool, attempts: usize, delay_ms: u64)
```

when those values form one concept.

A good API should make correct use easy and misuse difficult.

---

## Avoid hard to infer booleans

Avoid:

```rust
render(input, true, false)
```

when the call site cannot communicate what the booleans mean.

Prefer:

```rust
render(input, Color::Always, Links::Disabled)
```

Booleans are fine when naturally self-explanatory or named at construction:

```rust
Options {
    recursive: true,
}
```

---

## Keep visibility narrow

Default to private.

Use:

- `pub(super)` for tightly scoped module sharing,
- `pub(crate)` for crate-internal sharing,
- `pub` only for intentional external API.

Do not make something public merely to make tests or implementation easier.

Public API is a maintenance commitment.

---

## Follow standard Rust conventions

Use standard traits where they naturally express the operation:

- `From`,
- `TryFrom`,
- `AsRef`,
- `AsMut`,
- `Borrow`,
- `IntoIterator`,
- `FromIterator`,
- `Default`,
- `Display`,
- `FromStr`.

Follow conversion naming:

- `as_*` for cheap borrowed views,
- `to_*` for producing a new value and potentially allocating,
- `into_*` for ownership-consuming conversion.

Constructors should generally be:

- `new`,
- `with_*`,
- or a meaningful domain verb such as `open`, `connect`, `bind`, `parse`, `load`.

Do not invent custom conversion conventions when standard Rust already expresses the idea.

---

## Use generics for caller ergonomics, not abstraction theater

At a public boundary, an ergonomic signature such as:

```rust
fn open(path: impl AsRef<Path>)
```

can be appropriate.

Inside implementation code, normalize to simple concrete borrowed types when that improves
readability.

Do not make every helper generic because it can be.

Prefer concrete return types unless hiding the implementation type helps callers.

Use `impl Iterator` or `impl Future` where appropriate.

Avoid `Box<dyn Trait>` unless runtime polymorphism is genuinely required.

---

## Builders

Use a builder when construction has several optional or independently configurable values.

A good builder:

- has sensible defaults,
- keeps the common path concise,
- prevents invalid combinations where practical,
- validates at a clear boundary,
- uses consistent ownership semantics.

Do not create a builder when a constructor or options struct is clearer.

---

## Newtypes

Use newtypes when they add semantics or enforce an invariant.

Good examples:

```rust
struct ImageId(i64);
struct Similarity(f32);
struct ModelPath(PathBuf);
```

Do not introduce wrapper types merely to hide another type without gaining meaning or safety.

---

# 7. Error handling

## Use `thiserror`

Typed errors are required.

Use:

```rust
thiserror::Error
```

Do not introduce:

- `anyhow`,
- `eyre`,
- routine `Box<dyn std::error::Error>` erasure.

If an external API requires an erased error boundary, keep the rest of the application typed and
perform the erasure only at that boundary.

---

## `VisionGrepError` is the application-level error

`VisionGrepError` is the top-level application error type.

Focused subsystem error types may exist when they make a subsystem more cohesive or reusable.

For example:

```rust
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("failed to open image index at {path}")]
    Open {
        path: PathBuf,
        #[source]
        source: rusqlite::Error,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum VisionGrepError {
    #[error(transparent)]
    Index(#[from] IndexError),

    #[error(transparent)]
    Model(#[from] ModelError),
}
```

Do not split errors into many tiny enums merely for architectural aesthetics.

A single `VisionGrepError` is fine when it remains cohesive.

The requirement is **typed, meaningful errors**, not a specific number of enums.

---

## Add semantic context and preserve sources

Errors should describe what operation failed in VisionGrep's vocabulary.

Prefer:

```rust
#[error("failed to decode image at {path}")]
ImageDecode {
    path: PathBuf,
    #[source]
    source: image::ImageError,
}
```

over forwarding a low-level error without context.

Use:

- `#[source]` to preserve underlying errors,
- `#[from]` when conversion is unambiguous and loses no useful context,
- `#[error(transparent)]` when a wrapper intentionally adds no semantic context,
- `map_err` when translation adds domain context.

Never call `.to_string()` on an error merely to fit it into a `String` error variant.

Do not parse error messages.

---

## Prefer `?`

Use `?` for ordinary propagation.

Use explicit `match` when the code is actually:

- recovering,
- translating,
- falling back,
- retrying,
- logging,
- branching on error kind.

Avoid boilerplate matches that simply re-return the same error.

---

## `unwrap`, `expect`, and panics

Production code should not use `unwrap()`.

Prefer typed errors and `?`.

`expect()` should also be avoided in production code. If a structural invariant truly cannot fail
and cannot be better expressed through the type system, an `expect()` may be used only when:

- the invariant is local and immediately understandable,
- failure represents a programming bug rather than input/runtime failure,
- the message states the invariant.

Example:

```rust
root.expect("parser construction guarantees exactly one root node")
```

Never use `expect()` for:

- filesystem behavior,
- user input,
- database state,
- model loading,
- model inference,
- networking,
- serialization,
- configuration.

Do not use `panic!`, `unreachable!`, or `todo!` for ordinary runtime conditions.

Prefer proving impossible states through types and exhaustive matching.

---

# 8. Control flow

## Keep the happy path visible

Prefer:

- early returns,
- `?`,
- `let ... else`,
- exhaustive `match`,
- small meaningful helpers.

Avoid deep nesting.

Prefer:

```rust
let Some(image) = image else {
    return Ok(None);
};
```

when it keeps the main path unindented.

---

## Use `match` when cases matter

Use `match` for:

- enum state,
- state machines,
- multiple meaningful outcomes,
- places where exhaustiveness protects correctness.

Use `if let` when only one case matters.

Do not compress readable control flow into dense chains of:

- `map`,
- `and_then`,
- `filter`,
- `then_some`,
- `inspect`,

merely to save lines.

Intermediate locals are good when their names communicate domain meaning.

---

# 9. Functions, structs, and modules

## Functions should do one coherent thing

Split a function when doing so:

- names a useful concept,
- separates policy from mechanism,
- separates parsing from validation,
- separates validation from execution,
- removes real duplication,
- creates a meaningful test seam.

Do not split code into tiny helpers that merely rename syntax.

A sequential function may be long if splitting it would make the control flow harder to understand.

---

## Methods versus free functions

Put behavior on a type when the behavior:

- depends on that type's owned state,
- enforces the type's invariants,
- naturally belongs to the abstraction.

Prefer a free function for a pure/stateless transformation when no natural receiver exists.

For example, a `ModelSession` method that performs inference can be appropriate, while a pure
normalization helper does not need to be a method merely for stylistic consistency.

---

## Resource-owning types

Runtime resources such as:

- ONNX Runtime sessions,
- SQLite connections,
- model handles,
- caches with lifecycle,

should be owned by explicit types and released through RAII.

Do not use runtime global state, global service locators, or `lazy_static` / `once_cell` as a place to
hide runtime-owned resources.

Pass dependencies explicitly.

Compile-time constants and immutable static data are fine.

---

## Module structure is architectural, not historical

The current file/module layout is not sacred.

Refactor or split modules when doing so creates clearer:

- ownership,
- responsibilities,
- dependency direction,
- testability.

Do not preserve historical module boundaries merely because they already exist.

Do not create generic dumping grounds such as:

- `utils.rs`,
- `helpers.rs`,
- `common.rs`.

Put shared behavior in the most specific module that owns the concept, or create a clearly named
module for that concept.

Do not add files merely to reduce line count.

### Structure modules around cohesive concepts

A module should represent one coherent domain concept, subsystem, or architectural responsibility.

Good module boundaries often correspond to concepts such as:

- model loading and inference,
- embedding construction and normalization,
- indexing and persistence,
- search and ranking,
- CLI parsing,
- configuration,
- model download and installation,
- output formatting.

Do not group code merely because:

- it was written at the same time,
- it is used by the same command,
- it contains similar syntax,
- a file became "too long".

A module should have a meaningful reason to exist that can be stated in one short sentence.

If the explanation is "miscellaneous things used in several places," the boundary is probably wrong.

### Each invariant should have a clear owner

Important invariants should be enforced by the module or type that owns the concept.

Examples:

- the embedding layer owns the invariant that produced embeddings are normalized,
- the index layer owns persistence and schema invariants,
- the model layer owns model and session initialization invariants,
- the CLI layer owns argument parsing and user-facing exit behavior.

Do not require distant callers to remember subsystem invariants manually.

Prefer APIs where the owning module makes invalid behavior difficult or impossible.

### Keep dependency direction clean

Prefer dependencies that flow inward toward stable domain logic.

As a rule:

- CLI and presentation code may depend on application and domain code,
- orchestration may depend on model, index, and search subsystems,
- domain and search logic should not depend on CLI presentation details,
- persistence code should not depend on command-line parsing,
- model and inference code should not know about process exit codes,
- low-level modules should not import high-level application modules merely for convenience.

Avoid circular conceptual dependencies even when Rust's module system technically permits the code
to be arranged.

When two modules repeatedly depend on each other's internals, reconsider the boundary:

- move the shared concept to the module that actually owns it,
- extract a smaller neutral domain type or module,
- or merge the modules if they are really one cohesive subsystem.

Do not create abstraction layers solely to eliminate an import cycle. Fix the conceptual ownership.

### Keep public surfaces small

A module should expose the smallest API needed by its callers.

Prefer:

- private helpers,
- private implementation types,
- `pub(super)` and `pub(crate)` for deliberate internal sharing,
- a small set of public domain operations.

Do not expose internal structs or fields just so another module can reach through the abstraction.

Prefer asking the owning module to perform an operation:

```rust
index.insert(record)?;

over exposing the raw database connection so unrelated code can issue arbitrary SQL.

Prefer meaningful domain methods and result types over leaking implementation details.

### Separate orchestration from implementation

Top-level orchestration may coordinate several subsystems, but it should not absorb their internal
logic.

---

# 10. Traits and abstraction

Introduce a trait only when it provides a real benefit:

- multiple meaningful implementations,
- substitution at a real boundary,
- a generic algorithm,
- ecosystem integration,
- a useful test seam that cannot be achieved more simply.

Do not add a trait because another implementation might exist someday.

Concrete types are often the best abstraction.

Keep trait bounds near the methods or impls that need them.

Prefer associated types when an implementation has one natural item/output/error type.

---

# 11. Enums and state machines

Prefer enums over combinations of flags for mutually exclusive states.

Pattern-match explicitly on important internal enums.

Avoid catch-all handling such as:

```rust
match state {
    State::Ready => run(),
    _ => {}
}
```

when exhaustive matching would let the compiler catch future changes.

Use wildcard patterns intentionally for:

- external `#[non_exhaustive]` enums,
- genuinely irrelevant variants.

---

# 12. Paths, strings, and bytes

Filesystem paths are represented with:

- `&Path` for borrowed paths,
- `PathBuf` for owned paths.

Do not model filesystem paths as `String` or `&str`.

Do not assume paths are valid UTF-8.

Use `OsStr` / `OsString` for OS-native strings when needed.

Convert paths to text only at display or serialization boundaries where text is actually required.

Do not use lossy UTF-8 conversion unless lossy behavior is an explicit product decision.

For CLI behavior, preserve non-UTF-8 paths and arguments wherever the platform permits it.

---

# 13. VisionGrep embedding invariant

Embeddings use `Vec<f32>` unless a deliberate representation change is being made.

**Every embedding leaving the embedding layer must be L2-normalized.**

Never:

- persist an unnormalized embedding,
- compare an unnormalized embedding,
- require callers to remember to normalize an embedding manually.

Normalization belongs at the embedding boundary.

Functions that produce embeddings must return normalized embeddings.

Similarity and ranking code may rely on this invariant.

If the representation changes in the future, prefer a type that makes the invariant explicit, for
example a dedicated normalized embedding type, when doing so improves the API.

---

# 14. Collections and iteration

Choose collections according to semantics:

- `Vec<T>` for ordered contiguous values,
- `HashMap<K, V>` for keyed lookup,
- `HashSet<T>` for membership,
- `BTreeMap<K, V>` / `BTreeSet<T>` when ordering matters.

Do not use a map when a struct or enum is the real model.

Use iterator adapters for straightforward transformations.

Use loops when the logic is:

- stateful,
- branching,
- fallible,
- multi-step,
- easier to understand explicitly.

Do not collect an iterator merely to immediately iterate it.

Use `with_capacity` when a useful size estimate is already available and doing so remains clear.

---

# 15. Concurrency model

## VisionGrep is synchronous by default, not concurrency-free

Do **not** treat "synchronous CLI" as "all work must happen serially".

Concurrency is allowed when it makes startup or processing materially faster and remains simple.

Examples include:

- opening SQLite while the ORT/model session initializes,
- CPU-parallel image embedding,
- parallel independent filesystem work where appropriate.

The key question is which concurrency primitive matches the work.

---

## Prefer threads or Rayon for blocking/native work

SQLite opening, ONNX Runtime initialization, filesystem calls, and CPU-heavy inference are
blocking/native operations.

When overlapping a small number of independent blocking operations, prefer simple scoped threading
or Rayon rather than introducing an async runtime.

For example, conceptually:

```rust
let (index, model) = std::thread::scope(|scope| {
    let index = scope.spawn(|| ImageIndex::open(index_path));
    let model = ModelSession::load(model_config)?;

    let index = index
        .join()
        .map_err(|_| VisionGrepError::InitializationThreadPanicked)??;

    Ok::<_, VisionGrepError>((index, model))
})?;
```

or use `rayon::join` when Rayon is already the natural dependency.

Do not introduce `Arc<Mutex<Vec<_>>>` merely to gather parallel results.

Prefer data-parallel transforms and returned values:

```rust
let embeddings: Vec<_> = image_paths
    .par_iter()
    .map(|path| embed_image(path, &session))
    .collect::<Result<_, _>>()?;
```

when error semantics require all work to succeed.

Do not silently discard errors with `filter_map(... .ok())` unless skipping failures is an explicit
product behavior.

---

## Async Rust is allowed, but must earn its complexity

There is **no blanket ban on async Rust**.

Introduce async only when the workload materially benefits from asynchronous I/O concurrency or an
async ecosystem integration.

Reasonable examples may include:

- many concurrent network requests,
- remote model/artifact fetching where concurrency matters,
- an async protocol/client library that is clearly preferable to blocking alternatives,
- future service/server functionality,
- workflows with many independently waiting I/O operations.

Async is **not** justified merely because two blocking initialization operations can run at the same
time.

For a small number of blocking tasks, a scoped thread or Rayon is usually simpler.

Do not add Tokio or another async runtime merely to overlap:

- SQLite initialization,
- ORT session creation,
- a handful of filesystem operations,
- CPU-bound inference.

If async is introduced:

- keep synchronous domain/pure computation synchronous,
- isolate runtime-specific code near the application/I/O boundary,
- do not perform blocking ORT/SQLite/CPU-heavy work directly on executor threads,
- use the runtime's blocking facility when needed,
- do not hold locks across `.await`,
- bound concurrency,
- make cancellation and shutdown behavior explicit.

---

## Shared mutable state is a last resort

Do not reach reflexively for:

```rust
Arc<Mutex<T>>
```

Prefer:

- ownership,
- immutable shared state,
- returned values,
- Rayon reductions/collections,
- channels,
- narrower state.

Use shared mutable state only when the problem genuinely requires shared mutation.

---

# 16. CLI behavior

Treat CLI behavior as a public API.

Prefer typed `clap` derive structs and enums for normal CLI parsing.

Model mutually exclusive modes explicitly.

Separate:

- argument parsing,
- validation,
- orchestration,
- domain logic,
- presentation.

Use sensible defaults.

Do not require configuration for the common case merely to make the implementation generic.

---

## stdout and stderr contract

Normal search results go to **stdout**.

Progress indicators, warnings, and diagnostics go to **stderr**.

Never mix progress output with result output.

Preserve shell composability:

```text
visiongrep "query" ./photos/ | head -3
```

should emit only result data to the pipe.

Use `indicatif` for progress UI when appropriate.

Do not leave:

- `dbg!`,
- debug `println!`,
- temporary tracing,
- ad-hoc data dumps,

in production code.

---

## Exit codes

Preserve this CLI contract:

- `0` — command completed successfully with results,
- `1` — search completed successfully but no result met the threshold,
- `2` — operational error.

Only the top-level CLI/application boundary may call:

```rust
std::process::exit(...)
```

Subsystem and library code returns typed values/errors.

If the exit-code contract intentionally changes, update tests and user-facing documentation with the
same change.

---

# 17. Logging and diagnostics

Libraries and domain modules should not print to stdout/stderr as incidental side effects.

CLI presentation code owns user-facing output.

For diagnostics in more complex application flows, prefer structured tracing if it materially helps.

Do not add a logging/tracing framework solely for one debug message.

Error messages should contain enough semantic context to diagnose the failed operation.

---

# 18. Dependencies

Before adding a dependency:

1. check whether `std` solves the problem clearly,
2. check whether an existing dependency already solves it,
3. consider whether a small local implementation is genuinely simpler,
4. then add a dependency if it materially improves the solution.

Do not add dependencies for trivial syntax sugar.

Do not reimplement complex, security-sensitive, protocol-sensitive, or widely solved components
merely to avoid a good dependency.

Keep dependency features narrow and intentional.

Avoid enabling broad default feature sets without checking what they include.

Current preferred ecosystem choices:

- typed errors: `thiserror`,
- CLI parsing: `clap`,
- CPU/data parallelism: `rayon`,
- serialization: `serde` when serialization is required,
- HTTP services, if ever introduced: `axum` + Tower ecosystem,
- structured diagnostics when needed: `tracing`.

These are defaults, not reasons to add dependencies when they are unnecessary.

---

# 19. Performance

Write clear code first while remaining aware of allocation and algorithmic complexity.

Avoid obvious performance mistakes:

- repeated allocation in hot loops,
- repeated parsing of invariant data,
- accidental quadratic algorithms,
- unnecessary large clones,
- collecting large iterators without need,
- unbounded buffering,
- unnecessary synchronization,
- formatting strings that will never be used,
- serializing independent expensive initialization when simple concurrency would reduce startup.

For performance-sensitive changes:

- identify the hot path,
- measure when practical,
- use existing benchmarks,
- add focused benchmarks when a performance claim matters.

Do not introduce:

- unsafe code,
- elaborate caching,
- custom allocators,
- complicated concurrency,
- obscure micro-optimizations,

without evidence that the complexity is justified.

Prefer a simple measured optimization over a theoretically sophisticated one.

---

# 20. Unsafe Rust

Prefer safe Rust.

If a crate does not require unsafe code, prefer:

```rust
#![forbid(unsafe_code)]
```

for new crates/modules where that policy is practical.

Do not add `unsafe` merely to bypass the borrow checker.

If unsafe code is genuinely required:

- keep the unsafe region as small as possible,
- wrap it in a safe abstraction,
- document every invariant with `// SAFETY:`,
- reason explicitly about pointers, lengths, alignment, aliasing, lifetimes, and initialization,
- add focused tests.

Unsafe code requires stronger justification than ordinary code.

---

# 21. Macros

Prefer:

- functions,
- traits,
- generics,
- normal data structures,

over macros when they solve the problem cleanly.

Use `macro_rules!` when it removes substantial structural repetition.

Keep macro syntax familiar to Rust where practical.

Avoid procedural macros unless they provide a clear ergonomic benefit worth their compile-time and
debugging complexity.

Do not hide important side effects or surprising control flow in innocent-looking macros.

---

# 22. Documentation and comments

Comments explain **why**, not obvious syntax.

Use comments for:

- non-obvious invariants,
- safety reasoning,
- model/protocol constraints,
- compatibility requirements,
- performance tradeoffs,
- intentionally unusual implementation choices.

Bad:

```rust
// Increment count.
count += 1;
```

Good:

```rust
// Include the synthetic root so persisted node IDs remain stable.
count += 1;
```

Prefer clearer code and better names over comments explaining confusing implementation.

---

## Public API rustdoc

Public library API should document important:

- semantics,
- invariants,
- errors,
- panics,
- safety requirements,
- surprising allocation/complexity behavior,
- examples when usage is not obvious.

Do not add verbose rustdoc to trivial private helpers.

---

# 23. Naming

Use standard Rust naming conventions.

Choose names based on domain meaning.

Prefer precise names such as:

- `request_timeout`,
- `parse_config`,
- `ImageIndex`,
- `Similarity`,
- `is_empty`.

Avoid vague names such as:

- `thing`,
- `data`,
- `process`,
- `handle_stuff`,
- `do_it`.

Avoid redundant names when the containing type already supplies context:

```rust
struct Image {
    id: ImageId,
}
```

is usually better than:

```rust
struct Image {
    image_id: ImageId,
}
```

unless the longer name genuinely disambiguates something.

Short conventional names such as `i`, `tx`, `rx`, and `buf` are fine in small obvious scopes.

---

# 24. Testing

Tests should protect behavior and invariants, not implementation trivia.

For a bug fix, add a regression test that fails before the fix when practical.

Prefer:

- unit tests for local deterministic logic,
- integration tests for public behavior,
- table-driven cases when several cases express one rule,
- property tests when a broad invariant materially benefits from them.

Test meaningful:

- success cases,
- error cases,
- boundaries,
- empty inputs,
- malformed inputs,
- persistence behavior,
- compatibility behavior.

Do not over-mock.

Prefer lightweight real values, fakes, temporary files/databases, and in-memory implementations when
simpler than a mocking framework.

Do not make production APIs worse solely to simplify tests.

---

## VisionGrep test boundaries

Unit-test deterministic application logic such as:

- embedding normalization,
- similarity calculations,
- ranking,
- thresholding/filtering,
- parsing and validation,
- SQLite round-trips,
- index behavior,
- serialization/persistence invariants,
- CLI result formatting where practical.

Ordinary unit tests should not depend on:

- downloaded model files,
- expensive ONNX inference,
- external network services.

Model-backed behavior belongs in explicit integration tests when such coverage is useful.

Tests may use `unwrap()` / `expect()` when failure should immediately fail the test.

---

# 25. Cargo and feature design

Features should be:

- intentional,
- narrowly scoped,
- additive where practical.

Do not introduce feature combinations that silently alter unrelated semantics.

Optional dependencies should generally be tied to the feature that needs them.

Do not enable broad dependency feature sets without need.

Do not change the Rust edition, `rust-version`, or MSRV as incidental cleanup.

If the workspace declares a compatibility floor, preserve it unless the task explicitly changes it.

---

# 26. Serialization and persistence

Treat serialized and persisted formats as compatibility boundaries.

Use explicit types rather than ad-hoc maps when the schema is known.

Do not change:

- field names,
- enum representations,
- defaults,
- database semantics,
- compatibility behavior,

incidentally.

If a persisted/external format changes, make the migration and compatibility story explicit.

Do not expose internal implementation structures as external schemas merely because serialization is
easy.

---

# 27. Security and robustness

Treat external input as untrusted.

Validate appropriate:

- lengths,
- ranges,
- counts,
- paths,
- identifiers,
- model metadata,
- protocol fields,
- resource usage.

Avoid unbounded allocation based directly on attacker-controlled values.

Avoid unbounded recursion on external input.

Do not silently truncate or wrap numeric conversions.

Prefer fallible conversion:

```rust
usize::try_from(value)?
```

when narrowing may fail.

Do not use `as` for potentially lossy numeric conversions unless truncation/wrapping semantics are
intentional and obvious.

---

# 28. Strong anti-patterns

The following are not universally impossible, but their appearance should trigger scrutiny.

## Routine error erasure

Avoid:

```rust
fn run() -> Result<(), Box<dyn std::error::Error>>
```

Use typed `thiserror` errors.

---

## Borrow-checker escape cloning

Avoid:

```rust
let config = self.config.clone();
```

when the clone exists only to avoid reasoning about ownership.

---

## Premature dynamic abstraction

Avoid:

```rust
struct Service {
    backend: Box<dyn Backend>,
}
```

when there is only one meaningful backend and no runtime substitution requirement.

---

## Boolean blindness

Avoid:

```rust
execute(true, false, true);
```

---

## Stringly typed state

Avoid:

```rust
if state == "ready" {
    // ...
}
```

when the state has a closed set of values.

---

## Deeply nested happy paths

Avoid:

```rust
if let Some(a) = a {
    if let Ok(b) = make_b(a) {
        if b.is_valid() {
            // ...
        }
    }
}
```

when early exits make the main path clearer.

---

## Catch-all internal enum handling

Avoid:

```rust
match state {
    State::Ready => run(),
    _ => {}
}
```

when exhaustive matching would better protect future changes.

---

## Shared mutable state by default

Avoid reaching immediately for:

```rust
Arc<Mutex<T>>
```

Prefer ownership, immutable sharing, returned values, Rayon collection/reduction, or channels first.

---

## Async merely to express parallelism

Avoid introducing Tokio/async solely to run two independent blocking initialization routines
concurrently.

Use a scoped thread or Rayon for blocking work when that is the simpler model.

---

# 29. Codex operating rules

## Inspect before editing

Before changing code:

- read the relevant module,
- inspect the types involved,
- inspect nearby tests,
- search for related implementations,
- understand externally visible behavior.

Do not blindly copy an existing pattern.

Evaluate it against this file.

---

## Solve root causes

Do not paper over design problems with:

- gratuitous cloning,
- erased errors,
- broad lint allows,
- arbitrary sleeps,
- retries without policy,
- unnecessary locks,
- global mutable state,
- duplicated logic,
- unchecked conversions.

Fix the ownership, invariant, API, or layering problem at the appropriate level.

---

## Keep diffs focused

Do not:

- rename unrelated items,
- reorder unrelated code,
- rewrite entire modules solely for style,
- update unrelated dependencies,
- change public behavior unnecessarily.

Improving directly touched code is encouraged.

Repository-wide cleanup is not.

---

## Prefer compiler-enforced designs

When a small type change removes ambiguity, illegal states, or repeated checks, prefer the type
change.

Do not use advanced type machinery when a simple runtime check is clearer and equally safe.

---

## Treat heavyweight constructs as design decisions

Every nontrivial use of:

- `clone`,
- `Arc`,
- `Mutex`,
- `RwLock`,
- `Box<dyn Trait>`,
- `unsafe`,
- async,
- task spawning,
- global state,

must exist because its semantics are needed.

Do not use these constructs merely to silence compiler friction.

---

## Do not hide compiler feedback

Do not add:

- `.unwrap()`,
- unchecked indexing,
- lossy casts,
- broad `#[allow]`,
- `unsafe`,

merely to bypass compiler/linter feedback.

Fix the design where practical.

---

# 30. Required verification

For normal Rust changes, run:

```text
cargo fmt --all --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

For feature-gated changes, run the affected feature configurations.

Where compatible, include relevant `--all-features` verification.

For public documentation changes, run doctests when applicable.

For performance-sensitive changes, run relevant benchmarks when available.

If a command cannot be run:

- state exactly which command was not run,
- state why,
- do not imply verification succeeded.

---

# 31. Final review checklist

Before finishing any Rust change, verify:

- Is there a simpler design?
- Is the important behavior obvious from types and control flow?
- Does each important signature communicate its contract?
- Are invalid states represented explicitly?
- Is ownership natural?
- Are clones semantically justified?
- Are allocations intentional?
- Is shared ownership actually necessary?
- Is shared mutable state actually necessary?
- Are errors typed with `thiserror`?
- Do errors preserve useful sources?
- Did I avoid `anyhow` and routine error erasure?
- Is the happy path easy to read?
- Did I avoid premature traits, generics, builders, macros, async, or indirection?
- Are public APIs conventional and difficult to misuse?
- Are filesystem paths represented with `Path` / `PathBuf`?
- Are embeddings normalized at the embedding boundary?
- Are numeric conversions checked when loss is possible?
- Is concurrency bounded?
- Does the chosen concurrency model match the work?
- If async was introduced, was asynchronous I/O actually the reason?
- Are locks kept away from `.await`?
- Are stdout and stderr used according to the CLI contract?
- Are exit codes preserved?
- Are persistent/external formats changed only intentionally?
- Are relevant error and boundary cases tested?
- Is the diff focused?
- Does directly touched code meet this file even if old surrounding code does not?
- Did `cargo fmt --all --check` pass?
- Did `cargo check --workspace --all-targets` pass?
- Did Clippy pass with `-D warnings`?
- Did relevant tests pass?

If not, improve the code before considering the task complete.
