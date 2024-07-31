# Development Guide

## Introduction
This development guide provides an overview of the development model for the `experiment-nextgenjax` project. It includes information on the branching strategy, code review process, continuous integration practices, local development setup, testing, deployment, and contributing guidelines.

## Branching Strategy
The project follows a feature branching model to manage code integration. The main branches are:
- `main`: The stable branch containing the latest release version of the project.
- `develop`: The branch where the latest development changes are integrated.

Feature branches are created from the `develop` branch and are named using the following convention:
- `feature/<feature-name>`: For new features or enhancements.
- `bugfix/<bugfix-name>`: For bug fixes.
- `hotfix/<hotfix-name>`: For urgent fixes to the `main` branch.

## Code Review Process
Code reviews are an essential part of the development process to ensure code quality and maintainability. The steps for code reviews and pull requests are as follows:
1. Create a feature branch from the `develop` branch.
2. Implement the changes and commit them to the feature branch.
3. Push the feature branch to the remote repository.
4. Create a pull request (PR) from the feature branch to the `develop` branch.
5. Request a code review from one or more project maintainers.
6. Address any feedback or requested changes from the code review.
7. Once approved, the PR is merged into the `develop` branch.

## Continuous Integration
The project uses a CI/CD workflow to automate testing and ensure code quality. The CI/CD workflow is defined in the `.github/workflows/ci_cd_workflow.yml` file and includes the following steps:
- The workflow is triggered on push and pull request events.
- Jobs are set up for Ubuntu, Windows, and macOS environments using Python 3.8.
- The steps within each job include checking out the code, setting up Python, installing dependencies, and running tests with `pytest`.

## Local Development Setup
To set up a local development environment, follow these steps:
1. Clone the repository:
   ```
   git clone https://github.com/VishwamAI/experiment-nextgenjax.git
   cd experiment-nextgenjax
   ```
2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Testing
To run tests locally, use the following command:
```
pytest
```
The test results will be displayed in the terminal. To add new tests, create test files in the `tests` directory and follow the existing test structure.

## Deployment
If applicable, the deployment process for the application or library will be documented here. This may include steps for building, packaging, and releasing the project.

## Contributing
To contribute to the project, follow these guidelines:
1. Fork the repository and create a new branch for your changes.
2. Implement your changes and commit them to your branch.
3. Push your branch to your forked repository.
4. Create a pull request from your branch to the `develop` branch of the main repository.
5. Follow the code review process outlined above.

For any questions or assistance, contact the project maintainers.
