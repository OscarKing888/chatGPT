$tip = '0409:00000409'  # English (United States) - US keyboard
$list = Get-WinUserLanguageList

if (-not (($list | ForEach-Object { $_.InputMethodTips }) -contains $tip)) {
    $list[0].InputMethodTips.Add($tip)
    Set-WinUserLanguageList $list -Force
}

Set-WinDefaultInputMethodOverride -InputTip $tip

Start-Process "ms-settings:typing"

Get-WinDefaultInputMethodOverride
Get-WinUserLanguageList | Select-Object LanguageTag, InputMethodTips

#Set-WinDefaultInputMethodOverride
