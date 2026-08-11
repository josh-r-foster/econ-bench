# Model identifiers and path keys

Reviewed 2026-08-11

## Scope

EconBench uses a semantic model identifier and a derived model path key. They serve different purposes and must not be substituted for one another.

The semantic identifier is the `id` value in `config/models.json`. It appears in result metadata, release matrix cells, command arguments, dashboard labels, and model registries. It is never altered for display or scientific identity.

The path key is one portable filesystem and URL component derived from the semantic identifier. Canonical release paths and generated filenames use this key.

## Literal keys

An identifier remains unchanged when it begins with a lowercase ASCII letter or digit, contains only lowercase ASCII letters, digits, periods, underscores, and hyphens, does not end in a period, and does not use a reserved Windows device stem.

Every identifier registered for benchmark version `1.0.0` satisfies this rule. Existing committed filenames therefore remain unchanged.

## Encoded keys

Every other identifier is encoded as a tilde followed by the lowercase hexadecimal representation of its UTF-8 bytes. The tilde distinguishes encoded keys from literal keys.

| Model identifier | Path key |
| --- | --- |
| `gpt-4o` | `gpt-4o` |
| `gpt-5.2` | `gpt-5.2` |
| `a/b` | `~612f62` |
| `a_b` | `a_b` |
| `A` | `~41` |
| `con` | `~636f6e` |

The encoding preserves exact Unicode code points through their UTF-8 representation. It does not normalize case or Unicode spelling. Distinct semantic identifiers therefore remain distinct keys on case insensitive filesystems.

## Canonical decoding

A decoder accepts unchanged literal keys and encoded keys produced by the encoder. It rejects malformed hexadecimal text, invalid UTF-8, and encoded aliases for identifiers that qualify as literals.

Rejecting aliases ensures that one semantic identifier has exactly one path key. The manifest validator also checks that registered keys are unique.

## Implementations

Python producers and consumers use `src.results.model_ids.model_id_to_path_component`. Tools that recover semantic identifiers from directory names use `model_id_from_path_component`.

The static website uses `web/model_ids.js`. Python and browser implementations share the same acceptance vectors in the offline quality gate.

Ad hoc replacement of slashes or colons is prohibited. Underscore replacement is not reversible and can map identifiers containing a slash or colon to the same filename as `a_b`.

## Canonical paths

Path templates use the placeholder `model_key`. Result contents retain `metadata.model.id` as the semantic identifier.

Legacy prototype paths that contain only current registered identifiers already match the canonical key. An older underscore path for an unregistered identifier is ambiguous and must not be decoded by guesswork. Migration must obtain the semantic identifier from file contents, a manifest, or an explicit source mapping.
