Mark's Research Blog

# Useful commands:

Serving: `hugo serve`
Serving with drafts: `hugo serve -d`

Creation of this blog:
```
1. Install Hugo

sudo apt-get update
sudo apt-get install hugo

2. Clone github setup to this dir

mkdir markkurzeja.github.io
cd markkurzeja.github.io
git clone https://github.com/MarkKurzeja/markkurzeja.github.io.git

--or--

git remote add origin https://github.com/MarkKurzeja/markkurzeja.github.io.git

if you do not want to clone from github.

3. Make the site (note this will nuke everything else in the dir)

hugo new site . --force

4. Follow the quickstart: https://themes.gohugo.io/themes/beautifulhugo/

# Add the theme as a submodule in git
git submodule add https://github.com/halogenica/beautifulhugo.git themes/beautifulhugo
# Copy the content from the example site
cp -r themes/beautifulhugo/exampleSite/* . -iv

5. Run 

hugo serve











h
hugo
```
