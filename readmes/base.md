# Base
The base package includes only functions that have minimal external dependencies and are useful in a variety of situations such as parallelization and filesystem operations.

## Parallelization
### `embarrassingly_parallel`
This function allows you to call multiple functions in parallel by providing either positional arguments, keyword arguments, or both. The returns from these function calls are arranged in a list, ordered by the input arguments.

### `embarrassingly_parallel_simple`
Allows you to execute a list of functions (that take no arguments) in parallel and returns the results in the order the functions were provided.

## File System
### `create_file_dir`
Create a directory and its parent directories for a specified Path.