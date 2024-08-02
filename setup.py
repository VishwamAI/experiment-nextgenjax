from setuptools import setup, find_packages

setup(
    name='advancedjax',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        # Add other dependencies as needed
    ],
    description='AdvancedJax: State-of-the-Art AI and Machine Learning Framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='VishwamAI',
    author_email='example@example.com',  # Replace with actual email if available
    url='https://github.com/VishwamAI/experiment-nextgenjax',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)
