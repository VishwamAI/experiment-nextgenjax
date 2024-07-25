# Dependencies for AIPhoenix-SpeechTranscriber

The AIPhoenix-SpeechTranscriber component, inspired by the Whisper library, requires the following dependencies:

- numba: Provides just-in-time compilation for Python code, optimizing performance for numerical functions.
- numpy: A fundamental package for scientific computing with Python, providing support for large, multi-dimensional arrays and matrices.
- torch: An open-source machine learning library based on the Torch library, providing a flexible deep learning framework.
- tqdm: A fast, extensible progress bar for loops and or code blocks.
- more-itertools: A library that extends the built-in itertools library with additional functions and tools for working with iterables.
- tiktoken: (Dependency not found in standard repositories, may require further investigation or could be a custom or renamed package)
- triton: A language and compiler for writing highly efficient custom Deep Learning operations, with a specific version constraint for x86_64 Linux platforms (>=2.0.0, <3).

Note: The 'triton' dependency is specifically for x86_64 Linux platforms and may not be required or compatible with other platforms. Further investigation is needed to ensure cross-platform compatibility for the NextGenJax project.
