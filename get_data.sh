#!/bin/bash

# make sure curl is installed
if ! command -v curl &> /dev/null
then
    echo "curl could not be found"
    exit 1
fi

# generate data directory if it doesn't exist
mkdir -p data

# download data
curl -L https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json -o data/movies.json
