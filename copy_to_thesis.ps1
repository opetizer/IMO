chcp 65001 | Out-Null

$src = "D:\university\Research\IMO\output"
$dst = "D:\university\毕业设计\毕业论文\output"

Write-Host "Source: $src"
Write-Host "Destination: $dst"

# Clear destination
if (Test-Path $dst) {
    Write-Host "Clearing destination directory..."
    Remove-Item -Path "$dst\*" -Recurse -Force
} else {
    New-Item -ItemType Directory -Path $dst -Force | Out-Null
}

# Function to copy files recursively
function Copy-Files {
    param(
        [string]$s,
        [string]$d
    )

    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
    }

    # Copy files (png, html, txt, csv only)
    Get-ChildItem -Path $s -File | Where-Object {
        $_.Extension -in @('.png', '.html', '.txt', '.csv')
    } | ForEach-Object {
        $target = Join-Path $d $_.Name
        Copy-Item $_.FullName $target -Force
    }

    # Recursively copy subdirectories
    Get-ChildItem -Path $s -Directory | ForEach-Object {
        Copy-Files -s $_.FullName -d (Join-Path $d $_.Name)
    }
}

# Handle special mapping for cooccurrence graphs from meeting subdirectories
$committees = @('MEPC', 'MSC', 'CCC', 'SSE', 'ISWG-GHG')
foreach ($comm in $committees) {
    $commSrc = Join-Path $src $comm
    if (Test-Path $commSrc) {
        # Copy committee files
        Copy-Files -s $commSrc -d (Join-Path $dst $comm)

        # Handle cooccurrence graphs from meeting subdirectories
        $commDst = Join-Path $dst "$comm\cooccurrence"
        if (-not (Test-Path $commDst)) {
            New-Item -ItemType Directory -Path $commDst -Force | Out-Null
        }

        Get-ChildItem -Path $commSrc -Directory | ForEach-Object {
            $cooccFile = Join-Path $_.FullName "cooccurrence_graph.png"
            if (Test-Path $cooccFile) {
                $session = $_.Name
                # Extract session number like "77" from "MEPC 77"
                $sessionNum = $session -replace '[^\d]', ''
                if ($sessionNum -ne '') {
                    $targetName = "${comm}_${sessionNum}_cooccurrence.png"
                    $targetPath = Join-Path $commDst $targetName
                    Copy-Item $cooccFile $targetPath -Force
                    Write-Host "Copied cooccurrence: $session -> $targetName"
                }
            }
        }
    }
}

# Copy overall directories (stance_analysis, dynamic_analysis, deep_analysis, visualization)
$overallDirs = @('stance_analysis', 'dynamic_analysis', 'deep_analysis', 'visualization')
foreach ($dir in $overallDirs) {
    $srcDir = Join-Path $src $dir
    if (Test-Path $srcDir) {
        Copy-Files -s $srcDir -d (Join-Path $dst $dir)
    }
}

Write-Host "Done!"
