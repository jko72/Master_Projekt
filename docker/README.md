# Docker build and run
Prerequisites: You need a valid docker and docker compose installation.

## Command line must be within repository top dir
```bash
cd .../<dir>/docker
```
## Create custom .env-file (<ENV_NAME>.env, e.g. user.env)
Use [<template.env>](./template.env) to create your own matching env-file for correct user id, port and path forwarding. 

### 
```bash
# In docker-compose.yaml the namespace of <project_name> and <service_name> were defined as:
    # <project_name>: lidar_range 
    # <service_name>: dev

# make sure the .../docker/entrypoint.sh file is executable on host machine
sudo chmod +x ./entrypoint.sh

# Build docker image uncached
    # template: 
    # docker compose --env-file <ENV_NAME>.env -p <project_name> build --no-cache <service_name>
docker compose --env-file user.env -p lidar_range build --no-cache dev

# Create and start docker container in detached mode - explictly service 'dev' from source image of project lidar_range
    # template: 
    # docker compose --env-file <ENV_NAME>.env -p <project_name> up -d <service_name>
docker compose --env-file user.env -p lidar_range up -d dev

# Stop service
    # template: 
    # docker compose --env-file <ENV_NAME>.env -p <project_name> stop <service_name>
docker compose --env-file user.env -p lidar_range stop dev

# Stop all services and remove container
    # template: 
    # docker compose --env-file <ENV_NAME>.env -p <project_name> down
docker compose --env-file user.env -p lidar_range down
```
