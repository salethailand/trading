set -euo pipefail
git init
git config user.name "Your Name"
git config user.email "you@example.com"
printf ".venv/\n__pycache__/\n*.pyc\nrelease_bundle/*.tgz\n" > .gitignore
git add -A
git commit -m "Release: p39 equity (E0=100k) + proxy reports; bundle prepared"
git tag -a "release-$(date +%Y%m%d)-p39" -m "p39 release (equity fixed E0)"
# git remote add origin <your-remote-url>
# git push --follow-tags
