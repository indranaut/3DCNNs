#!/usr/bin/env bash

echo "Compiling Profo files."

protoc -I=. --python_out=. ./protos/*.proto

echo "Compilation finished."
