#!/bin/bash
podman run -ti --rm -v "$(pwd)":/root/shared -w /root/shared jhale/fenics-error-estimation:latest "sudo -i /bin/bash -l"
