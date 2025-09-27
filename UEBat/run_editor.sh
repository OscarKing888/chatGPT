#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
/Volumes/USB_2T/UE561/LocalBuilds/Engine/Mac/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor -project="$SCRIPT_DIR/SoccerBallGame.uproject"