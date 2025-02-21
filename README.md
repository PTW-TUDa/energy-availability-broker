# Project Title

Insert a short project description here

## Usage

__When using this template it is imperative to switch ALL mentions of "energy_information_service" in the configuration to the name you are actually going to call your project. Also rename the "energy_information_service" folder to your chosen name.__

Here you should give a short introduction on how to use your project

## Development
If you want to perform development work, you have to install Poetry. See the [eta-utility docs](https://eta-utility.readthedocs.io/en/master/guide/development.html) for more information.

### Using internal connectors
If you want to use the eta-utility internal connectors, ask your supervisor for the wheel (.whl) file and copy it to a folder called "wheels" inside the project folder.
Then you have to add it via Poetry:

    poetry add "./wheels/eta_utility_internal_connectors-0.1b2-py3-none-any.whl"

### Installation
To install the project along with its development dependencies, execute the following command:

    poetry install --extras develop --sync

Followed by

    poetry run pre-commit install

After this you are ready to perform the first commits to the repository.

Pre-commit ensures that the repository accepts your commit, automatically fixes some code styling problems and provides some hints for better coding.

### Adding dependencies
Adding dependencies to the project can be done via

    poetry add <package-name>@latest

followed by

    poetry install --extras develop --sync
