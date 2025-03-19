# Project Title

implement the energy information service for the eta factory

## Usage

just put 'poetry run fastapi run ./energy_information_service/main.py' into the terminal

## Development

to run a dev session, put 'poetry run fastapi dev ./energy_information_service/main.py'

If you want to perform development work, you have to install Poetry. See the [eta-utility docs](https://eta-utility.readthedocs.io/en/master/guide/development.html) for more information.


### Installation
To install the project along with its development dependencies, execute the following command:

    poetry install --sync

Followed by

    poetry run pre-commit install

After this you are ready to perform the first commits to the repository.

Pre-commit ensures that the repository accepts your commit, automatically fixes some code styling problems and provides some hints for better coding.

### Adding dependencies
Adding dependencies to the project can be done via

    poetry add <package-name>@latest
