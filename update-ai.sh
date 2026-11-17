#!/bin/bash

echo ""
if command -v "claude" >/dev/null 2>&1; then
	echo "### update claude code ###"
	claude update
else
	echo "claud not installed"
fi

echo ""
if command -v "codex" >/dev/null 2>&1; then
	echo "### update codex ###"
	codex update
else
	echo "codex not installed"
fi

echo ""
if command -v "agy" >/dev/null 2>&1; then
	echo "### update antigravity cli ###"
	agy update
else
	echo "agy not install"
fi

echo ""
if command -v "opencode2" >/dev/null 2>&1; then
	echo "### update opencode v2 ###"
	opencode2 upgrade
else
	echo "opencode2 not install"
fi

echo ""
if command -v "omp" >/dev/null 2>&1; then
	echo "### update omp ###"
	omp update
else
	echo "omp not install"
fi

