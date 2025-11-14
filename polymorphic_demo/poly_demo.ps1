# Polymorphic Demo - PowerShell Script
# This demonstrates polymorphic behavior in PowerShell
# EDUCATIONAL PURPOSE ONLY - No malicious behavior

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*59) -ForegroundColor Cyan
Write-Host "POLYMORPHIC DEMONSTRATION - POWERSHELL" -ForegroundColor Yellow
Write-Host "Educational Purpose Only - No Malicious Behavior" -ForegroundColor Green
Write-Host ("="*60) -ForegroundColor Cyan

# Generate random mutation ID
$mutationId = Get-Random -Minimum 1000 -Maximum 9999
Write-Host "[PS DEMO] Mutation ID: $mutationId" -ForegroundColor Cyan

# Generate random junk data to change file signature
$junkSize = Get-Random -Minimum 50 -Maximum 200
$junkData = -join ((65..90) + (97..122) | Get-Random -Count $junkSize | ForEach-Object {[char]$_})
Write-Host "[PS DEMO] Junk Data Length: $($junkData.Length)" -ForegroundColor Cyan

# Calculate hash of current state
$dataToHash = "$mutationId$junkData"
$hash = [System.Security.Cryptography.SHA256]::Create().ComputeHash([System.Text.Encoding]::UTF8.GetBytes($dataToHash))
$hashString = [System.BitConverter]::ToString($hash).Replace("-", "").Substring(0, 16)
Write-Host "[PS DEMO] Current Hash: $hashString..." -ForegroundColor Cyan

# Simulate polymorphic encryption
function Invoke-SimpleXOR {
    param(
        [string]$text,
        [int]$key
    )
    $encrypted = ""
    foreach ($char in $text.ToCharArray()) {
        $encrypted += [char]([int]$char -bxor $key)
    }
    return $encrypted
}

$encryptionKey = Get-Random -Minimum 1 -Maximum 255
$payload = "This is a benign demonstration payload"
Write-Host "`n[PS DEMO] Original Payload: $payload" -ForegroundColor Green
Write-Host "[PS DEMO] Encryption Key: $encryptionKey" -ForegroundColor Cyan

$encrypted = Invoke-SimpleXOR -text $payload -key $encryptionKey
$encryptedBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($encrypted))
Write-Host "[PS DEMO] Encrypted: $($encryptedBase64.Substring(0, 30))..." -ForegroundColor Yellow

$decrypted = Invoke-SimpleXOR -text $encrypted -key $encryptionKey
Write-Host "[PS DEMO] Decrypted: $decrypted" -ForegroundColor Green

# Simulate code obfuscation with random execution paths
Write-Host "`n[PS DEMO] Demonstrating Random Execution Paths" -ForegroundColor Magenta

function Invoke-PathA {
    Write-Host "[PS DEMO] Executing Path A" -ForegroundColor DarkCyan
    Write-Host "[PS DEMO] Benign Operation: Current Date = $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor White
}

function Invoke-PathB {
    Write-Host "[PS DEMO] Executing Path B" -ForegroundColor DarkCyan
    Write-Host "[PS DEMO] Benign Operation: PowerShell Version = $($PSVersionTable.PSVersion)" -ForegroundColor White
}

function Invoke-PathC {
    Write-Host "[PS DEMO] Executing Path C" -ForegroundColor DarkCyan
    Write-Host "[PS DEMO] Benign Operation: Computer Name = $env:COMPUTERNAME" -ForegroundColor White
}

# Randomly choose execution path
$paths = @('A', 'B', 'C')
$chosenPath = Get-Random -InputObject $paths

switch ($chosenPath) {
    'A' { Invoke-PathA }
    'B' { Invoke-PathB }
    'C' { Invoke-PathC }
}

# Add junk operations
Write-Host "`n[PS DEMO] Adding Junk Operations..." -ForegroundColor DarkYellow
for ($i = 0; $i -lt 3; $i++) {
    $junkCalc = (Get-Random -Minimum 1 -Maximum 1000) * (Get-Random -Minimum 1 -Maximum 1000)
    Start-Sleep -Milliseconds 10
}
Write-Host "[PS DEMO] Junk operations completed" -ForegroundColor DarkYellow

# Demonstrate mutation capability
Write-Host "`n[PS DEMO] Demonstrating Mutation Capability" -ForegroundColor Magenta
$newMutationId = Get-Random -Minimum 1000 -Maximum 9999
$newJunkSize = Get-Random -Minimum 50 -Maximum 200
$newJunkData = -join ((65..90) + (97..122) | Get-Random -Count $newJunkSize | ForEach-Object {[char]$_})

$newDataToHash = "$newMutationId$newJunkData"
$newHash = [System.Security.Cryptography.SHA256]::Create().ComputeHash([System.Text.Encoding]::UTF8.GetBytes($newDataToHash))
$newHashString = [System.BitConverter]::ToString($newHash).Replace("-", "").Replace("-", "").Substring(0, 16)

Write-Host "[PS DEMO] Old Hash: $hashString..." -ForegroundColor Yellow
Write-Host "[PS DEMO] New Hash: $newHashString..." -ForegroundColor Yellow
Write-Host "[PS DEMO] Hashes are different: $($hashString -ne $newHashString)" -ForegroundColor Green

Write-Host "`n[PS DEMO] Demonstration Complete" -ForegroundColor Green
Write-Host ("="*60) -ForegroundColor Cyan

