
# Elasticsearch Log Analysis Project

This project demonstrates how to use the **Elastic Stack** (Elasticsearch, Logstash, Kibana) to analyze web server access logs, as part of a data processing and log analysis challenge.

## 📄 Contents

- `data_science_elastic.pdf`: Final project submission with screenshots and instructions.
- `elasticsearch_complete_guide.pdf`: A complete beginner-friendly guide on Elasticsearch installation, data ingestion, querying, and visualization.

## 📌 Project Description

The project simulates a real-world scenario where you work for an e-commerce company (e.g., BASF) and are tasked with analyzing HTTP web server logs. The goal is to gain insights about user access and error requests using Elasticsearch tools.

### Key Steps:
1. Load logs using **Logstash**
2. Filter HTTP responses different from 200 using **Kibana DevTools**
3. Build a dashboard in **Kibana** to visualize status codes and request counts

## 🧪 How to Run

### 1. Install Elasticsearch and Kibana

```bash
# Elasticsearch
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.7.0-linux-x86_64.tar.gz
tar -xzf elasticsearch-8.7.0-linux-x86_64.tar.gz
cd elasticsearch-8.7.0
./bin/elasticsearch

# Kibana (in a separate terminal)
wget https://artifacts.elastic.co/downloads/kibana/kibana-8.7.0-linux-x86_64.tar.gz
tar -xzf kibana-8.7.0-linux-x86_64.tar.gz
cd kibana-8.7.0
./bin/kibana
```

### 2. Load Logs with Logstash

Use the following Logstash config (`logstash.conf`) to parse Apache logs:

```conf
input {
  file {
    path => "/path/to/access.log"
    start_position => "beginning"
  }
}
filter {
  grok {
    match => { "message" => "%{COMBINEDAPACHELOG}" }
  }
}
output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "weblogs"
  }
}
```

Then run:

```bash
logstash -f logstash.conf
```

### 3. Use Kibana to Explore Data

- Open [http://localhost:5601](http://localhost:5601)
- Use **Discover** to inspect data
- Create visualizations and dashboards
- Use **DevTools** to run queries like:

```json
GET /weblogs/_search
{
  "query": {
    "bool": {
      "must_not": {
        "match": {
          "response": "200"
        }
      }
    }
  }
}
```

## 🧠 Useful Links

- [Elasticsearch Docs](https://www.elastic.co/guide/en/elasticsearch/reference/index.html)
- [Kibana Docs](https://www.elastic.co/guide/en/kibana/index.html)
- [Logstash Docs](https://www.elastic.co/guide/en/logstash/index.html)

---

If you're new to Elasticsearch, check out the full guide in `elasticsearch_complete_guide.pdf`.
