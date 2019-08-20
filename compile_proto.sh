#!/usr/bin/env bash


echo "Compiling proto files."

protoc -I=. --python_out=. ./protos/*.proto

echo "Finished compiling proto files."