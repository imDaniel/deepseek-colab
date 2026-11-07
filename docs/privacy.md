# Repository privacy

Publish source, credential-free configuration snapshots and sanitized benchmark results. Keep runtime state, OAuth credentials, API/management keys, SSH private keys, private infrastructure identifiers, account balances and raw operational logs outside Git. The default `state/` and local `experiments/` directories are ignored; a custom `LLM_STATE` should also be outside the repository or explicitly ignored.

The dated [configuration snapshot](../configs/colab-optimized-20261107.json) contains only settings and boot parameters. It does not contain an API key, management key, OAuth token, host address or user filesystem path. Runtime source remains authoritative; this snapshot is a record, not an automatically loaded override.

Before publishing, review staged paths and run Gitleaks on both the staged change and all reachable history:

```sh
git diff --cached --name-only
gitleaks git --staged --redact --no-banner .
gitleaks git --log-opts="--all --full-history" --redact --no-banner .
```

The scanner configuration extends the default rules and allows only the exact public environment-variable name `LLAMA_DSV41_IDX_CHUNK_MIB`, which otherwise resembles a generic key. It does not exclude tests or benchmark files.

Secret scanning does not establish absence of all personal information. Separately review commit author/committer identities, annotated tag identities, commit messages, release notes and attached archives. Use the repository owner's GitHub no-reply identity for new commits. Check `.gitignore` with `git check-ignore` and ensure sensitive files were never tracked: ignoring a file does not remove it from earlier commits.

Configure the no-reply identity locally in this repository and set `user.useConfigOnly=true` so Git does not guess an address from the machine account. Before pushing, verify **both** the author and committer of the actual new commit use `@users.noreply.github.com`; author environment variables and hosting APIs can override local Git settings. An API publishing path is suitable only when it explicitly preserves both identities. If a connector cannot set those fields, publish the verified local commit through Git instead of accepting the connector's default identity. Keep personal addresses out of public diagnostics when checking metadata. These requirements concern commit metadata; retain legitimate upstream source and license attribution.

Confirmed exposed credentials require revocation or rotation as well as removal. Removing historical personal information requires an explicitly reviewed history rewrite, including affected tags. Old GitHub commit views, cached archives and external copies may require separate cleanup; rewriting branch history alone does not erase every previously public copy. Private scan output and recovery bundles belong in ignored local storage, not in a public audit attachment.
