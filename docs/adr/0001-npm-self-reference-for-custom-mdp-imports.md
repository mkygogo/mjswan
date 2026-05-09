# npm self-reference for custom MDP source imports

Custom MDP Sources need to import mjswan base classes (`ObservationBase`, `CommandTerm`, etc.) without using relative paths that break outside the template directory. We added `mjswan` as a `devDependency` of itself in `package.json` (npm self-reference) and defined subpath exports (`mjswan/observation`, `mjswan/command`, `mjswan/math`, etc.) pointing at TypeScript source files inside `src/core/`. At build time, Vite resolves these imports through the self-reference symlink in `node_modules/mjswan`; external users who `npm install -D mjswan` get the same resolution with full IDE IntelliSense.

## Considered options

- **Vite alias in `_build_client.py`**: Inject `resolve.alias` dynamically at build time. Rejected because it requires the build client to maintain a parallel mapping of every public subpath, duplicating the `exports` field and diverging when new subpaths are added.
- **Separate `@mjswan/types` package**: Publish stub type declarations independently. Rejected because it adds a third package to keep in sync with PyPI and npm releases, with no benefit over self-reference since the template is already published as `mjswan`.
- **Relative imports (status quo)**: Fragile — only resolve because `_build_client.py` inlines files into the template directory. Breaks IDE IntelliSense entirely for authors outside the repo.

## Consequences

- `src/core/` must be included in the npm `files` array so subpath targets are present in the published package.
- Subpath exports point to `.ts` source, not compiled `.d.ts`, which requires consumers to use a bundler that handles TypeScript (Vite does; plain `tsc` does not).
- Adding a new public base class requires both a new subpath in `exports` and updating `src/core/` — two places, but both in the same file (`package.json`) and directory.
