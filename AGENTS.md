description: Use this rule when using ThreadRng crate

When using `ThreadRng` module
1. the `rand::thread_rng` method has been deprecated and renamed `rng`. USE `rng`
2. the `rand::Rng::gen_range` method is deprecated and renamed `randon_range`. USE `randon_range`

## Testing

- ALWAYS write tests for new code.
- ALWAYS run the tests after every code change and make sure the tests pass.
