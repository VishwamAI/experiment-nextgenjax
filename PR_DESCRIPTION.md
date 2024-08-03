# Fix Import Errors and Ensure All Tests Pass

## Summary of Changes
This pull request addresses several issues related to import errors and test failures in the NextGenJax project. The following changes have been made:

1. Updated package versions in `requirements.txt` for compatibility:
   - Adjusted `numpy` version range to resolve dependency conflicts
   - Updated `jax` and `jaxlib` to latest versions
   - Specified compatible versions for `chex`, `optax`, and other dependencies

2. Modified GitHub Actions workflow:
   - Added explicit installation steps for `optax` and `dm-haiku`
   - Included a step to list installed packages for debugging purposes
   - Removed the `deactivate` command from the CI workflow to prevent environment issues

3. Fixed import statements in test files:
   - Updated import paths to use the correct module structure
   - Ensured consistency across all test files

4. Addressed ModuleNotFoundError:
   - Added `fairscale` to the requirements to resolve missing module issues

5. Improved test suite:
   - Ensured all tests are discoverable and runnable by pytest
   - Fixed any broken test cases

## Detailed Changes
- Updated `requirements.txt` with compatible package versions
- Modified `.github/workflows/ci_cd_workflow.yml` to improve the CI/CD process
- Updated import statements in test files under `src/tests/`
- Added missing dependencies to resolve ModuleNotFoundErrors
- Fixed and improved test cases to ensure all tests pass

## Testing
All tests have been run locally and pass successfully. The GitHub Actions workflows have been updated to reflect these changes and should now complete without errors.

## Next Steps
- Review the changes and provide feedback
- Merge the pull request if all checks pass and changes are approved
- Monitor the project for any further issues related to imports or test failures

Link to Devin run: https://preview.devin.ai/devin/0c3b3947742f424083edd2db7ee1a676
