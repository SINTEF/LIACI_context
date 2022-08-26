#!/bin/ash

# The URL is the first argument
url=$1

# Print an error and exit if $url is not set
if [ -z "$url" ]; then
    echo "ERROR: $0 requires a URL as the first argument"
    exit 1
fi

# Read the file /data/url if it exists and compare it to $url
# If they are equal, print "URL is up to date" and exit
if [ -f /data/url ] ; then
  if [ "$(cat /data/url)" == "$url" ]; then
    echo "URL is up to date"
    exit 0
  fi
fi

# Remove everything in /data, this is outdated
rm -vrf /data/*

# Create a temporary file
tempfile=$(mktemp)

# Download the URL to a temporary file using curl
curl "$url" > "$tempfile"

# Unzip the file into data folder
unzip -o "$tempfile" -d /data

# find the first folder in /data
folder=$(find /data -mindepth 1 -maxdepth 1 -type d)

# Move everything from subfolders into the parent folder
mv -v $folder/* /data/

# Remove the folder
rmdir $folder

# Move the folder to its download location
mv "$tempfile" /data/data.zip

# Write the url in the /data/url file
echo "$url" > /data/url