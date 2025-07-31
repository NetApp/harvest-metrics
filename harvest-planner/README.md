# Harvest Planner

Harvest Planner estimates how much memory each poller needs to monitor ONTAP and StorageGRID clusters.
Here's how to use it:

1. Run the following Harvest command to gather object counts from you cluster(s)
   `bin/harvest planner -p poller` # one cluster 
   `bin/harvest planner`           # multiple clusters
   `bin/harvest planner --docker`  # multiple clusters and run the following Docker command for you

The planner command will create a `objects.json` file that contains the object counts for each cluster.

2. Run the following Docker command to estimate how much memory each poller needs to monitor its cluster.

```bash
docker run --rm \
  --volume "$(pwd)/objects.json:/objects.json" \
  ghcr.io/netapp/harvest-planner \
  estimate-memory -i /objects.json
```

# Development

Harvest-planner is written in Python and uses [Rye](https://rye.astral.sh/) for development.

To get started, install Rye, clone the repo, cd into `harvest-metrics/harvest-planner` and run the following command:

```bash
rye sync
```