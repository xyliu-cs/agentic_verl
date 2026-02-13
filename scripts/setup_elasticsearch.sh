#!/bin/bash
set -u
set -e

# Download Elasticsearch with the compatible version
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.29-linux-x86_64.tar.gz
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.29-linux-x86_64.tar.gz.sha512
shasum -a 512 -c elasticsearch-7.17.29-linux-x86_64.tar.gz.sha512

# Copy and extract Elasticsearch
mv elasticsearch-7.17.29-linux-x86_64.tar.gz /home/elasticsearch-7.17.29-linux-x86_64.tar.gz && cd /home
tar -xzf elasticsearch-7.17.29-linux-x86_64.tar.gz
cd elasticsearch-7.17.29/

# Create elastic user and set ownership
sudo useradd -m -s /bin/bash elastic || true  # ignore if user already exists
sudo chown -R elastic:elastic /home/elasticsearch-7.17.29

# Start Elasticsearch as elastic user
sudo -u elastic /home/elasticsearch-7.17.29/bin/elasticsearch &

# Wait for Elasticsearch to start
sleep 10

# Test connection
curl -X GET "localhost:9200/"
