# Contributing to tokenizers-clj

Thanks for your interest in improving `tokenizers-clj`. Bug reports, fixes, and
focused feature contributions are all welcome.

## Before you start

- For anything beyond a trivial fix, **open an issue first**. This lets us agree
  on the approach before you invest time.
- Check existing issues and pull requests to avoid duplicate work.

## Development

This is a Clojure library. You need a JDK and
[Leiningen](https://leiningen.org/). A project that moved to `deps.edn` uses the
Clojure CLI instead: see the README.

```bash
lein test     # run the test suite
lein check    # AOT-compile; must be free of reflection warnings
```

Requirements for a mergeable change:

- **Tests first.** Add or update tests for the behavior you change. For a bug
  fix, include a regression test that fails before your fix and passes after it.
- **Green build.** `lein test` passes and `lein check` reports **zero**
  reflection warnings.
- **No scope creep.** Keep each pull request to one logical change.

## Commits and pull requests

- Follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:` …).
- Keep the subject in the imperative mood and under ~72 characters.
- Update `CHANGELOG.md` when your change is user-visible.
- Rebase on the latest `main` before you open the pull request.

## License

If you contribute, you agree to license your contributions under the same
license as this project. See `LICENSE` or the README.
