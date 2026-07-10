Get-ChildItem -Recurse -Filter *.mp4 | ForEach-Object {
    $tmp = $_.FullName.Replace(".mp4","_tmp.mp4")

    ffmpeg -y -i $_.FullName -vf fps=1 -an $tmp

    if ($LASTEXITCODE -eq 0) {
        Move-Item -Force $tmp $_.FullName
    }
}