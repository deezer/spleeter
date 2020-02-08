#!/bin/bash
# tldr --update removes symlinks

# create symlink tldr entry
ln -s ~/Applications/spleeter/spleeter/tldr_spleeter.md /Users/vortamir/.tldrc/tldr-master/pages/osx/spleeter.md

echo 'tldr entry created'

tldr spleeter