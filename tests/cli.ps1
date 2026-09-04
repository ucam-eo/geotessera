# GeoTessera CLI Tests for Windows (PowerShell)
# Equivalent to tests/cli.t cram tests

param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

# Track test results
$script:TestsPassed = 0
$script:TestsFailed = 0
$script:TestErrors = @()

function Write-TestHeader {
    param([string]$Title)
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
}

function Write-TestResult {
    param(
        [string]$TestName,
        [bool]$Passed,
        [string]$Details = ""
    )
    if ($Passed) {
        Write-Host "[PASS] $TestName" -ForegroundColor Green
        $script:TestsPassed++
    } else {
        Write-Host "[FAIL] $TestName" -ForegroundColor Red
        if ($Details) {
            Write-Host "       $Details" -ForegroundColor Red
        }
        $script:TestsFailed++
        $script:TestErrors += $TestName
    }
}

function Invoke-Geotessera {
    param(
        [string[]]$Arguments
    )
    $cmdLine = "geotessera $($Arguments -join ' ')"
    Write-Host "Executing: $cmdLine" -ForegroundColor DarkGray
    $output = & geotessera @Arguments 2>&1
    return $output
}

function Show-DirectoryTree {
    param(
        [string]$Path,
        [string]$Indent = "",
        [int]$Depth = 0
    )

    if (-not (Test-Path $Path)) {
        Write-Host "${Indent}[Directory does not exist: $Path]" -ForegroundColor Yellow
        return
    }

    $item = Get-Item $Path
    if ($Depth -eq 0) {
        Write-Host "$($item.Name)/" -ForegroundColor Cyan
    }

    $children = Get-ChildItem -Path $Path -ErrorAction SilentlyContinue | Sort-Object { -not $_.PSIsContainer }, Name
    $childCount = $children.Count
    $index = 0

    foreach ($child in $children) {
        $index++
        $isLast = ($index -eq $childCount)
        $prefix = if ($isLast) { "└── " } else { "├── " }
        $nextIndent = $Indent + $(if ($isLast) { "    " } else { "│   " })

        if ($child.PSIsContainer) {
            Write-Host "${Indent}${prefix}$($child.Name)/" -ForegroundColor Blue
            # Recurse into subdirectory (full depth)
            Show-DirectoryTree -Path $child.FullName -Indent $nextIndent -Depth ($Depth + 1)
        } else {
            $size = "{0:N0}" -f $child.Length
            Write-Host "${Indent}${prefix}$($child.Name) ($size bytes)" -ForegroundColor White
        }
    }
}

# Setup
Write-TestHeader "Setup"

# Disable fancy terminal output
$env:TERM = "dumb"

# Create temporary directories for test outputs and cache
$TestDir = Join-Path $env:TEMP "geotessera_test_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $TestDir -Force | Out-Null
Write-Host "Test directory: $TestDir"

$CacheDir = Join-Path $TestDir "cache"
New-Item -ItemType Directory -Path $CacheDir -Force | Out-Null
$env:XDG_CACHE_HOME = $CacheDir
Write-Host "Cache directory: $CacheDir"

try {
    # Test: Version Command
    Write-TestHeader "Test: Version Command"

    $versionOutput = Invoke-Geotessera -Arguments @("version")
    $versionOutput = ($versionOutput | Out-String).Trim()

    $expectedVersion = "0.10.2"
    $versionPassed = $versionOutput -eq $expectedVersion
    Write-TestResult -TestName "Version command returns $expectedVersion" -Passed $versionPassed -Details "Got: $versionOutput"

    if ($Verbose) {
        Write-Host "Output: $versionOutput"
    }

    # Test: Info Command (Library Info)
    Write-TestHeader "Test: Info Command (Library Info)"

    $infoOutput = Invoke-Geotessera -Arguments @("info", "--dataset-version", "v1")
    $infoString = $infoOutput | Out-String

    $hasAvailableYears = $infoString -match "Available years"
    Write-TestResult -TestName "Info command shows 'Available years'" -Passed $hasAvailableYears

    # Manifest replaces the old per-version registry.parquet (see PR #250).
    # Either of these phrases is acceptable — a cold run downloads the
    # manifest fresh; a warm cache logs "Using cached manifest" instead.
    $hasManifestFetch = $infoString -match "(Downloading|Using cached) manifest"
    Write-TestResult -TestName "Info command fetches manifest" -Passed $hasManifestFetch

    if ($Verbose) {
        Write-Host "Output:"
        Write-Host $infoString
    }

    # Test: Download Dry Run for UK Tile
    Write-TestHeader "Test: Download Dry Run for UK Tile"

    $dryRunOutput = Invoke-Geotessera -Arguments @(
        "download",
        "--bbox", "-0.1,51.3,0.1,51.5",
        "--year", "2024",
        "--format", "tiff",
        "--dry-run",
        "--dataset-version", "v1"
    )
    $dryRunString = $dryRunOutput | Out-String

    $hasFormat = $dryRunString -match "Format:\s+TIFF"
    Write-TestResult -TestName "Dry run shows Format: TIFF" -Passed $hasFormat

    $hasYear = $dryRunString -match "Year:\s+2024"
    Write-TestResult -TestName "Dry run shows Year: 2024" -Passed $hasYear

    $hasCompression = $dryRunString -match "Compression:\s+lzw"
    Write-TestResult -TestName "Dry run shows Compression: lzw" -Passed $hasCompression

    $hasDatasetVersion = $dryRunString -match "Dataset version:\s+v1"
    Write-TestResult -TestName "Dry run shows Dataset version: v1" -Passed $hasDatasetVersion

    $hasFoundTiles = $dryRunString -match "Found 16 tiles"
    Write-TestResult -TestName "Dry run finds 16 tiles" -Passed $hasFoundTiles

    $hasFilesToDownload = $dryRunString -match "Files to download:\s+16"
    Write-TestResult -TestName "Dry run shows Files to download: 16" -Passed $hasFilesToDownload

    if ($Verbose) {
        Write-Host "Output:"
        Write-Host $dryRunString
    }

    # Test: Download Single UK Tile (TIFF format)
    Write-TestHeader "Test: Download Single UK Tile (TIFF format)"

    $tiffOutputDir = Join-Path $TestDir "uk_tiles_tiff"

    $downloadTiffOutput = Invoke-Geotessera -Arguments @(
        "download",
        "--bbox", "-0.1,51.3,0.1,51.5",
        "--year", "2024",
        "--format", "tiff",
        "--output", $tiffOutputDir,
        "--dataset-version", "v1"
    )
    $downloadTiffString = $downloadTiffOutput | Out-String

    $hasSuccess = $downloadTiffString -match "SUCCESS.*16.*GeoTIFF"
    Write-TestResult -TestName "TIFF download shows SUCCESS with 16 files" -Passed $hasSuccess

    # Verify TIFF files were created
    $tiffSearchPath = Join-Path $tiffOutputDir "global_0.1_degree_representation\2024"
    $tiffFiles = @()
    if (Test-Path $tiffSearchPath) {
        $tiffFiles = Get-ChildItem -Path $tiffSearchPath -Filter "*.tif*" -Recurse -ErrorAction SilentlyContinue
    }

    $tiffFilesCreated = $tiffFiles.Count -gt 0
    Write-TestResult -TestName "TIFF files created" -Passed $tiffFilesCreated -Details "Found $($tiffFiles.Count) files"

    $tiffFileCount = $tiffFiles.Count -eq 16
    Write-TestResult -TestName "16 TIFF files created" -Passed $tiffFileCount -Details "Found $($tiffFiles.Count) files"

    if ($Verbose) {
        Write-Host "Output:"
        Write-Host $downloadTiffString
        Write-Host "TIFF files found: $($tiffFiles.Count)"
    }

    # Always show directory tree for debugging
    Write-Host ""
    Write-Host "Directory tree for TIFF download:" -ForegroundColor Magenta
    Show-DirectoryTree -Path $tiffOutputDir

    # Test: Download Single UK Tile (NPY format)
    Write-TestHeader "Test: Download Single UK Tile (NPY format)"

    $npyOutputDir = Join-Path $TestDir "uk_tiles_npy"

    $downloadNpyOutput = Invoke-Geotessera -Arguments @(
        "download",
        "--bbox", "-0.1,51.3,0.1,51.5",
        "--year", "2024",
        "--format", "npy",
        "--output", $npyOutputDir,
        "--dataset-version", "v1"
    )
    $downloadNpyString = $downloadNpyOutput | Out-String

    $hasNpySuccess = $downloadNpyString -match "SUCCESS.*Downloaded.*16 tiles"
    Write-TestResult -TestName "NPY download shows SUCCESS with 16 tiles" -Passed $hasNpySuccess

    # Verify NPY directory structure was created
    $npyEmbeddingsDir = Join-Path $npyOutputDir "global_0.1_degree_representation\2024"
    $embeddingsDirCreated = Test-Path $npyEmbeddingsDir
    Write-TestResult -TestName "Embeddings directory created" -Passed $embeddingsDirCreated

    $npyLandmasksDir = Join-Path $npyOutputDir "global_0.1_degree_tiff_all"
    $landmasksDirCreated = Test-Path $npyLandmasksDir
    Write-TestResult -TestName "Landmasks directory created" -Passed $landmasksDirCreated

    # Verify NPY files exist
    $npyFiles = @()
    if (Test-Path $npyEmbeddingsDir) {
        $npyFiles = Get-ChildItem -Path $npyEmbeddingsDir -Filter "*.npy" -Recurse -ErrorAction SilentlyContinue
    }

    $embeddingNpyCreated = ($npyFiles | Where-Object { $_.Name -match "grid_.*\.npy$" -and $_.Name -notmatch "_scales\.npy$" }).Count -gt 0
    Write-TestResult -TestName "Embedding NPY files created" -Passed $embeddingNpyCreated

    $npyFileCount = $npyFiles.Count -eq 32
    Write-TestResult -TestName "32 NPY files created" -Passed $npyFileCount -Details "Found $($npyFiles.Count) files"

    $scalesNpyCreated = ($npyFiles | Where-Object { $_.Name -match "_scales\.npy$" }).Count -gt 0
    Write-TestResult -TestName "Scales NPY files created" -Passed $scalesNpyCreated

    # Verify landmask TIFF files
    $landmaskTiffs = @()
    if (Test-Path $npyLandmasksDir) {
        $landmaskTiffs = Get-ChildItem -Path $npyLandmasksDir -Filter "*.tif*" -Recurse -ErrorAction SilentlyContinue
    }
    $landmaskTiffCreated = $landmaskTiffs.Count -gt 0
    Write-TestResult -TestName "Landmask TIFF files created" -Passed $landmaskTiffCreated -Details "Found $($landmaskTiffs.Count) files"

    if ($Verbose) {
        Write-Host "Output:"
        Write-Host $downloadNpyString
        Write-Host "NPY files found: $($npyFiles.Count)"
        Write-Host "Landmask TIFFs found: $($landmaskTiffs.Count)"
    }

    # Always show directory tree for debugging
    Write-Host ""
    Write-Host "Directory tree for NPY download:" -ForegroundColor Magenta
    Show-DirectoryTree -Path $npyOutputDir

    # Test: Info Command on Downloaded TIFF Tiles
    Write-TestHeader "Test: Info Command on Downloaded TIFF Tiles"

    $infoTiffOutput = Invoke-Geotessera -Arguments @("info", "--tiles", $tiffOutputDir)
    $infoTiffString = $infoTiffOutput | Out-String

    $hasTotalTiles = $infoTiffString -match "Total tiles:\s*16"
    Write-TestResult -TestName "Info shows Total tiles: 16" -Passed $hasTotalTiles

    $hasFormatInfo = $infoTiffString -match "Format:.*GEOTIFF|NPY|ZARR"
    Write-TestResult -TestName "Info shows format information" -Passed $hasFormatInfo

    $hasYearsInfo = $infoTiffString -match "Years:\s*2024"
    Write-TestResult -TestName "Info shows Years: 2024" -Passed $hasYearsInfo

    $hasCrsInfo = $infoTiffString -match "CRS:.*EPSG"
    Write-TestResult -TestName "Info shows CRS information" -Passed $hasCrsInfo

    $hasBandCount = $infoTiffString -match "128 bands"
    Write-TestResult -TestName "Info shows 128 bands" -Passed $hasBandCount

    if ($Verbose) {
        Write-Host "Output:"
        Write-Host $infoTiffString
    }

    # Test: Info Command on Downloaded NPY Tiles
    Write-TestHeader "Test: Info Command on Downloaded NPY Tiles"

    $infoNpyOutput = Invoke-Geotessera -Arguments @("info", "--tiles", $npyOutputDir)
    $infoNpyString = $infoNpyOutput | Out-String

    $hasNpyTotalTiles = $infoNpyString -match "Total tiles:\s*16"
    Write-TestResult -TestName "Info shows Total tiles: 16" -Passed $hasNpyTotalTiles

    $hasNpyFormat = $infoNpyString -match "Format:.*NPY"
    Write-TestResult -TestName "Info shows NPY format" -Passed $hasNpyFormat

    $hasNpyYears = $infoNpyString -match "Years:\s*2024"
    Write-TestResult -TestName "Info shows Years: 2024" -Passed $hasNpyYears

    $hasNpyCrs = $infoNpyString -match "CRS:.*EPSG"
    Write-TestResult -TestName "Info shows CRS information" -Passed $hasNpyCrs

    $hasNpyBands = $infoNpyString -match "128 bands"
    Write-TestResult -TestName "Info shows 128 bands" -Passed $hasNpyBands

    if ($Verbose) {
        Write-Host "Output:"
        Write-Host $infoNpyString
    }

    # Test: Resume Capability for NPY Downloads
    Write-TestHeader "Test: Resume Capability for NPY Downloads"

    $resumeOutput = Invoke-Geotessera -Arguments @(
        "download",
        "--bbox", "-0.1,51.3,0.1,51.5",
        "--year", "2024",
        "--format", "npy",
        "--output", $npyOutputDir,
        "--dataset-version", "v1"
    )
    $resumeString = $resumeOutput | Out-String

    $hasSkipped = $resumeString -match "Skipped.*48.*existing files|resume capability"
    Write-TestResult -TestName "Resume skips existing files" -Passed $hasSkipped

    if ($Verbose) {
        Write-Host "Output:"
        Write-Host $resumeString
    }

    # Test: Coverage Command
    Write-TestHeader "Test: Coverage Command"

    $coverageOutputDir = Join-Path $TestDir "coverage_output"
    New-Item -ItemType Directory -Force -Path $coverageOutputDir | Out-Null
    $coveragePng = Join-Path $coverageOutputDir "uk_coverage.png"

    $coverageOutput = Invoke-Geotessera -Arguments @(
        "coverage",
        "--country", "United Kingdom",
        "--output", $coveragePng,
        "--dataset-version", "v1"
    )
    $coverageString = $coverageOutput | Out-String

    # Check if PNG file was created
    $pngCreated = Test-Path $coveragePng
    Write-TestResult -TestName "Coverage PNG file created" -Passed $pngCreated

    # Check if JSON file was created (same directory, coverage.json)
    $coverageJson = Join-Path $coverageOutputDir "coverage.json"
    $jsonCreated = Test-Path $coverageJson
    Write-TestResult -TestName "Coverage JSON file created" -Passed $jsonCreated

    # Check if globe HTML was created
    $globeHtml = Join-Path $coverageOutputDir "globe.html"
    $htmlCreated = Test-Path $globeHtml
    Write-TestResult -TestName "Coverage globe.html created" -Passed $htmlCreated

    # Verify globe.html content is valid UTF-8 and contains expected elements
    if ($htmlCreated) {
        try {
            $globeContent = Get-Content -Path $globeHtml -Raw -Encoding UTF8
            $hasUtf8Charset = $globeContent -match '<meta charset="UTF-8">'
            Write-TestResult -TestName "globe.html has UTF-8 charset" -Passed $hasUtf8Charset

            $hasGlobeTitle = $globeContent -match "GeoTessera Globe Visualization"
            Write-TestResult -TestName "globe.html has correct title" -Passed $hasGlobeTitle

            # PR #250 replaced the static "Legend:" block with a dynamic
            # per-dataset "Layers" panel populated client-side from
            # coverage.json. Assert on the new anchor instead.
            $hasLayerPanel = $globeContent -match '<div id="layerList"'
            Write-TestResult -TestName "globe.html has layers panel" -Passed $hasLayerPanel
        } catch {
            Write-TestResult -TestName "globe.html is readable as UTF-8" -Passed $false -Details $_.Exception.Message
        }
    }

    if ($Verbose) {
        Write-Host "Output:"
        Write-Host $coverageString
        if ($pngCreated) {
            $pngSize = (Get-Item $coveragePng).Length
            Write-Host "PNG size: $pngSize bytes"
        }
        if ($htmlCreated) {
            $htmlSize = (Get-Item $globeHtml).Length
            Write-Host "globe.html size: $htmlSize bytes"
        }
    }

} catch {
    Write-Host ""
    Write-Host "ERROR: Test execution failed" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    $script:TestsFailed++
} finally {
    # Summary
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host "Test Summary" -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host "Passed: $script:TestsPassed" -ForegroundColor Green
    Write-Host "Failed: $script:TestsFailed" -ForegroundColor $(if ($script:TestsFailed -gt 0) { "Red" } else { "Green" })

    if ($script:TestErrors.Count -gt 0) {
        Write-Host ""
        Write-Host "Failed tests:" -ForegroundColor Red
        foreach ($error in $script:TestErrors) {
            Write-Host "  - $error" -ForegroundColor Red
        }
    }

    # Cleanup option
    Write-Host ""
    Write-Host "Test directory: $TestDir"
    Write-Host "To clean up test files, run: Remove-Item -Recurse -Force '$TestDir'"

    # Exit with appropriate code
    if ($script:TestsFailed -gt 0) {
        exit 1
    }
    exit 0
}
