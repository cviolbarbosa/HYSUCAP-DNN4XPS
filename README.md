# Deep Neural Networ for XPS

This repository hosts the source code for the proof-of-concept algorithm that uses DNN to extract information of x-ray photoemission spectra.
This information is useful for posterior correlation with EELS spectra. 


## Running application

To run this algorithm you need to install Docker: https://docs.docker.com/get-docker/ .

After docker is installed run the command:

```
docker-compose up

```

Open the browser and type `http://localhost:8889/notebooks/DNN4XPS.ipynb#`


## Modify and building yor own docker images

If you want to modify the source code and run it, use:

```
docker-compose -f docker-compose-dev.yml up

```
