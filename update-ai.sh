#!/bin/bash

if command -v "claude" >/dev/null 2>&1; then
	claude update
else
	echo "claud not installed"
fi


if command -v "codex" >/dev/null 2>&1; then
	npm i -g @openai/codex
else
	echo "codex not installed"
fi

if command -v "opencode" >/dev/null 2>&1; then
	opencode upgrade
else
	echo "opencode not install"
fi
