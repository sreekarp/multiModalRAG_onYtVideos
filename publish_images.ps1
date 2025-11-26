# Docker Hub Username
$USERNAME = "ninja19"

# Image Names
$BACKEND_IMAGE = "$USERNAME/ytrag-backend"
$FRONTEND_IMAGE = "$USERNAME/ytrag-frontend"

Write-Host "🚀 Starting Docker Build & Push Process..." -ForegroundColor Cyan

# 1. Login to Docker Hub
Write-Host " Logging in to Docker Hub..." -ForegroundColor Yellow
docker login
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker login failed. Please check your credentials."
    exit 1
}

# 2. Build Backend
Write-Host " Building Backend Image ($BACKEND_IMAGE)..." -ForegroundColor Yellow
Push-Location backend
docker buildx build --load -t "$BACKEND_IMAGE:latest" "${PWD}.Path"
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }
Pop-Location

# 3. Push Backend
Write-Host " Pushing Backend Image..." -ForegroundColor Yellow
docker push "$BACKEND_IMAGE:latest"
if ($LASTEXITCODE -ne 0) { exit 1 }

# 4. Build Frontend
Write-Host " Building Frontend Image ($FRONTEND_IMAGE)..." -ForegroundColor Yellow
Push-Location frontend
docker buildx build --load -t "$FRONTEND_IMAGE:latest" "${PWD}.Path"
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }
Pop-Location

# 5. Push Frontend
Write-Host " Pushing Frontend Image..." -ForegroundColor Yellow
docker push "$FRONTEND_IMAGE:latest"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "✅ All images built and pushed successfully!" -ForegroundColor Green
Write-Host "Backend:  $BACKEND_IMAGE:latest"
Write-Host "Frontend: $FRONTEND_IMAGE:latest"
