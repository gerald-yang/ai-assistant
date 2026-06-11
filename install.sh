#!/bin/bash

if command -v "claude" >/dev/null 2>&1; then
	echo "claude code already install"
else
	echo "install claude code"
	curl -fsSL https://claude.ai/install.sh | bash
	echo "alias clauded='claude --dangerously-skip-permissions'" >> ~/.bashrc
fi

if command -v "codex" >/dev/null 2>&1; then
	echo "codex is already installed"
else
	echo "install codex"
	curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
	export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
        [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
	nvm install --lts
	npm i -g @openai/codex
	echo "alias codexd='codex --dangerously-bypass-approvals-and-sandbox'" >> ~/.bashrc
fi

if command -v "agy" >/dev/null 2>&1; then
        echo "antigravity cli is already installed"
else
        echo "install antigravity cli"
	curl -fsSL https://antigravity.google/cli/install.sh | bash
	echo "alias agyd='agy --dangerously-skip-permissions'" >> ~/.bashrc
fi

if command -v "opencode" >/dev/null 2>&1; then
        echo "opencode is already installed"
else
        read -p "Do you want to install opencode? (y/N): " response
        case "$response" in
                [yY][eE][sS]|[yY])
                        echo "install opencode"
                        curl -fsSL https://opencode.ai/install | bash
                        ;;
                *)
                        echo "Skipping opencode installation."
                        ;;
        esac
fi

