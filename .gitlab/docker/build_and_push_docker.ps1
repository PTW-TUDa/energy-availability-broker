$ErrorActionPreference = "Stop"

# Resolve repository root from script location
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
Set-Location $repoRoot

# Load variables from .env (simple KEY=VALUE lines)
$envFile = Join-Path $repoRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]*)=(.*)$") {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"').Trim("'")
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
} else {
    Write-Host "ERROR: .env file not found at $envFile"
    exit 1
}

# Define variables
$PYTHON_VERSIONS = @("3.11")
$POETRY_VERSION = "2.3.2"
$DOCKERFILE_PATH = ".gitlab/docker/Dockerfile"
$REGISTRY_URL = "git-reg.ptw.maschinenbau.tu-darmstadt.de"
$IMAGE_PATH = "eta-fabrik/projekte/energy-availability-broker"

if (-not $env:DEPLOY_TOKEN) {
    Write-Host "ERROR: DEPLOY_TOKEN is not set. Please check your .env file."
    exit 1
}

# Docker login using GitLab deploy token
$env:DEPLOY_TOKEN | docker login $REGISTRY_URL --username PRIVATE-TOKEN --password-stdin
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to log in to the Docker registry."
    exit 1
}

foreach ($PYTHON_VERSION in $PYTHON_VERSIONS) {
    $IMAGE_NAME = "$REGISTRY_URL/$IMAGE_PATH/poetry$POETRY_VERSION`:py$PYTHON_VERSION"

    Write-Host "Building and pushing Docker image for Python $PYTHON_VERSION..."

    docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH `
        --build-arg PYTHON_VERSION=$PYTHON_VERSION `
        --build-arg POETRY_VERSION=$POETRY_VERSION `
        .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to build Docker image for Python $PYTHON_VERSION"
        exit 1
    }

    docker push $IMAGE_NAME
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to push Docker image for Python $PYTHON_VERSION"
        exit 1
    }

    Write-Host "Finished building and pushing Docker image for Python $PYTHON_VERSION!"
}

Write-Host "All Docker images built and pushed successfully!"
