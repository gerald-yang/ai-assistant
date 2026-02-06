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
	sudo snap install node --classic
	sudo npm i -g @openai/codex
fi

if command -v "opencode" >/dev/null 2>&1; then
        echo "opencode is already installed"
else
        echo "install opencode"
        curl -fsSL https://opencode.ai/install | bash
fi
