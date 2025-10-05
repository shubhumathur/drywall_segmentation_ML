# Set depth here (for all subfolders)
$maxDepth = Read-Host "Enter maximum depth (e.g., 3, 4)"
if (-not ($maxDepth -as [int])) { 
    Write-Host "Invalid depth, defaulting to 3" 
    $maxDepth = 3 
}

# Folders to ignore
$ignoreList = @("venv", "__pycache__")

# Function to show tree
function Show-Tree($path, $currentDepth) {
    if ($currentDepth -gt [int]$maxDepth) { return }

    # Get files and folders
    $files = Get-ChildItem -Path $path -File | Sort-Object Name
    $folders = Get-ChildItem -Path $path -Directory | Where-Object { $ignoreList -notcontains $_.Name } | Sort-Object Name

    # Print files first (optional)
    foreach ($file in $files) {
        $indent = " " * (($currentDepth - 1) * 4)
        Write-Host "$indent- $($file.Name)"
    }

    # Print folders and recurse
    foreach ($folder in $folders) {
        $indent = " " * (($currentDepth - 1) * 4)
        Write-Host "$indent- $($folder.Name)"
        Show-Tree -path $folder.FullName -currentDepth ($currentDepth + 1)
    }
}

# Start
$rootPath = Get-Location
Write-Host "Project tree for: $($rootPath.Path)`n"
Show-Tree -path $rootPath.Path -currentDepth 1
Write-Host "`nDone!"



<#
==============================================================
HOW TO RUN THIS SCRIPT
==============================================================

1. Save this script as "show_tree.ps1" in the folder you want to analyze.

2. Open PowerShell and navigate to the folder containing the script:
   Example:
       cd C:\Users\renus\drywall_segmentation

3. Run the script using:
       .\show_tree.ps1

4. The script will ask you for the maximum depth:
       Enter maximum depth (e.g., 3, 4)
   - Enter a number for how deep you want to see the folder tree.
   - Folders in the $ignoreList (venv, __pycache__) will be skipped automatically.

5. The folder tree will be printed in the PowerShell window with indentation representing depth.

==============================================================
NOTES
==============================================================
- Make sure your execution policy allows running scripts:
      Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
- You can edit the $ignoreList at the top of the script to skip other folders if needed.
#>