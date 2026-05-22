#!/bin/bash

if command -v "claude" >/dev/null 2>&1; then
	echo "claude code already install"
else
	echo "install claude code"
	curl -fsSL https://claude.ai/install.sh | bash
fi

if command -v "codex" >/dev/null 2>&1; then
	echo "codex is already installed"
else
	echo "install codex"
	curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
	source ~/.bashrc
	nvm install --lts
	npm i -g @openai/codex
fi

if command -v "opencode" >/dev/null 2>&1; then
        echo "opencode is already installed"
else
        echo "install opencode"
        curl -fsSL https://opencode.ai/install | bash
fi

if command -v "agy" >/dev/null 2>&1; then
        echo "antigravity cli is already installed"
else
        echo "install antigravity cli"
	curl -fsSL https://antigravity.google/cli/install.sh | bash
fi
