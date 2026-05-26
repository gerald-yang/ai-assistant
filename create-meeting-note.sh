#!/bin/bash

if [ -z "$1" ]; then
	echo "no input"
	exit -1
fi

ffmpeg -i "$1" out.mp3
whisper out.mp3

