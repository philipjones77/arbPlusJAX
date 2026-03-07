param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsFromCaller
)

# Windows PowerShell wrapper
python tools/run_validation.py @ArgsFromCaller
