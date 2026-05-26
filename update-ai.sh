#!/bin/bash

if command -v "claude" >/dev/null 2>&1; then
	echo "update claude code"
	claude update
else
	echo "claud not installed"
fi

if command -v "codex" >/dev/null 2>&1; then
	echo "update codex"
	npm i -g @openai/codex
else
	echo "codex not installed"
fi

if command -v "opencode" >/dev/null 2>&1; then
	echo "update opencode"
	opencode upgrade
else
	echo "opencode not install"
fi

if command -v "agy" >/dev/null 2>&1; then
	echo "update antigravity cli"
	agy update
else
	echo "agy not install"
fi
